"""
learning_cbf_reference.py -- reference implementations for the learning-based
sections of the CBF report (Section 8).

Unlike cbf_core.py / cbf_examples.py, this file needs PyTorch (and, for the
RL wrapper, a Gym-style environment).  It is written to be lifted into a real
project, not executed in a notebook sandbox:

    1. NeuralCBF            -- a neural barrier with the three training losses
    2. train_neural_cbf     -- counterexample-guided (CEGIS-style) training loop
    3. verification hooks   -- where a sound bound-propagation check plugs in
    4. CBFSafetyFilter      -- differentiable CBF-QP layer usable inside a policy
    5. SafetyFilterWrapper  -- Gym wrapper: filter actions during RL training
    6. cbf_penalty_reward   -- the "internalise the constraint" reward term

References for each construction are given in the docstrings and in Section 11
of the report.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:                       # pragma: no cover
    raise SystemExit("this reference file requires PyTorch: pip install torch") from exc


# ---------------------------------------------------------------------------
# 1. A neural barrier function
# ---------------------------------------------------------------------------


class NeuralCBF(nn.Module):
    """h_theta : R^n -> R, a neural candidate barrier.

    Design notes
    ------------
    * Smooth activations (tanh / softplus) keep grad h well defined; ReLU nets
      are also used and admit *exact* MILP/SMT verification (Zhang, Wu,
      Vorobeychik & Clark, NeurIPS 2023) at the price of non-smoothness.
    * Output is left unbounded: the sign of h carries the safety semantics.
    * For multi-agent problems, replace the MLP by a graph neural network whose
      receptive field is the sensing radius -- this is the GCBF construction
      (Zhang, So, Garg & Fan, T-RO 2025) and is what makes one certificate
      valid for any number of agents.
    """

    def __init__(self, dim_x: int, hidden: Tuple[int, ...] = (64, 64)):
        super().__init__()
        layers, last = [], dim_x
        for w in hidden:
            layers += [nn.Linear(last, w), nn.Tanh()]
            last = w
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """grad_x h(x) by autograd -- needed by every CBF loss term."""
        x = x.clone().requires_grad_(True)
        h = self.forward(x)
        (g,) = torch.autograd.grad(h.sum(), x, create_graph=True)
        return g


# ---------------------------------------------------------------------------
# 2. The three losses, and a counterexample-guided training loop
# ---------------------------------------------------------------------------


def cbf_losses(h_net: NeuralCBF,
               x_safe: torch.Tensor,
               x_unsafe: torch.Tensor,
               x_domain: torch.Tensor,
               f: Callable[[torch.Tensor], torch.Tensor],
               g: Callable[[torch.Tensor], torch.Tensor],
               u_max: float,
               alpha: Callable[[torch.Tensor], torch.Tensor],
               margin: float = 0.1):
    """The standard trio of neural-certificate losses.

        L_safe    : h(x) >= +margin on sampled safe states
        L_unsafe  : h(x) <= -margin on sampled unsafe states
        L_descent : sup_{||u||<=u_max} [ Lf h + Lg h u ] + alpha(h) >= margin

    The supremum over a norm ball is available in closed form, which is why the
    descent loss needs no inner optimisation:

        sup_u Lg h(x) u = u_max * ||grad h(x)^T g(x)||_2 .

    This is the loss structure shared by Dawson/Gao/Fan (T-RO 2023),
    So et al. (ICRA 2024) and the GCBF+ training framework.
    """
    L_safe = torch.relu(margin - h_net(x_safe)).mean()
    L_unsafe = torch.relu(margin + h_net(x_unsafe)).mean()

    gx = h_net.grad(x_domain)                                # (B, n)
    h = h_net(x_domain)                                      # (B,)
    Lfh = (gx * f(x_domain)).sum(-1)                         # (B,)
    LgT = torch.einsum("bn,bnm->bm", gx, g(x_domain))        # (B, m)
    sup = Lfh + u_max * LgT.norm(dim=-1)
    L_descent = torch.relu(margin - (sup + alpha(h))).mean()

    return L_safe, L_unsafe, L_descent


def train_neural_cbf(h_net, sampler, f, g, u_max, alpha,
                     steps: int = 3000, lr: float = 1e-3,
                     weights=(1.0, 1.0, 2.0), verifier=None):
    """Counterexample-guided synthesis (the CEGIS pattern).

    Loop:  train on the current dataset  ->  call a *sound* verifier  ->
    append the counterexamples it returns  ->  train again.  Without the
    verifier in the loop, a low training loss says nothing about the
    uncountably many states never sampled; with it, termination yields a
    certificate rather than a hope.

    ``verifier(h_net) -> (ok: bool, counterexamples: Tensor)`` is expected to
    be a bound-propagation / SMT / MILP procedure -- see ``verify_hint`` below.
    """
    opt = torch.optim.Adam(h_net.parameters(), lr=lr)
    ws, wu, wd = weights
    for k in range(steps):
        x_safe, x_unsafe, x_domain = sampler()
        Ls, Lu, Ld = cbf_losses(h_net, x_safe, x_unsafe, x_domain,
                                f, g, u_max, alpha)
        loss = ws * Ls + wu * Lu + wd * Ld
        opt.zero_grad()
        loss.backward()
        opt.step()

        if verifier is not None and (k + 1) % 500 == 0:
            ok, cex = verifier(h_net)
            if ok:
                print(f"[{k+1}] verified")
                return h_net, True
            sampler.add_counterexamples(cex)      # shrink the violation region
    return h_net, False


def verify_hint():
    """Where a sound check goes, and what the practical options are.

    * Linear bound propagation (CROWN / alpha-beta-CROWN family) over the
      network *and* its Jacobian: scalable, incomplete but sound; see
      Hu, Yang, Wei & Liu (CoRL 2024) for symbolic derivative bounds and the
      2025 work on scalable neural-CBF verification by bound propagation.
    * Exact MILP / SMT encodings for ReLU barriers: complete, exponential
      worst case (Zhang et al., NeurIPS 2023).
    * Branch-and-bound verification *inside* the training loop, so the
      certificate and the network are produced together (Wang, Knoedler,
      Mathiesen & Alonso-Mora, ECC 2024).
    * SOS programming when h and the dynamics are polynomial: the classical
      route, and still the cleanest when the plant fits.

    Whatever the tool: verify the *descent condition over the whole domain*,
    not just the sign of h on samples.  A learned zero level set that merely
    matches the obstacle geometry is a learned safe set, not a CBF.
    """
    raise NotImplementedError("plug in your verifier of choice")


# ---------------------------------------------------------------------------
# 3. A differentiable CBF-QP layer (BarrierNet-style)
# ---------------------------------------------------------------------------


class CBFSafetyFilter(nn.Module):
    """Differentiable projection onto the CBF half-space.

    For a single constraint the QP

        min ||u - u_nom||^2   s.t.  a(x) . u + b(x) >= 0

    has the closed-form solution

        u* = u_nom + a * max(0, -(a . u_nom + b)) / ||a||^2 ,

    which is differentiable almost everywhere and needs no QP solver -- the
    trick used to make multi-constraint safe RL cheap via a log-sum-exp
    aggregate barrier.  For genuinely multi-constraint problems, swap this for
    a differentiable QP layer (``cvxpylayers``/``qpth``) as in BarrierNet
    (Xiao et al., T-RO 2023); the gradients then come from the KKT conditions,
    and the class-K parameters themselves become trainable.
    """

    def __init__(self, h_fn, grad_h_fn, f_fn, g_fn, gamma: float = 1.0):
        super().__init__()
        self.h_fn, self.grad_h_fn = h_fn, grad_h_fn
        self.f_fn, self.g_fn = f_fn, g_fn
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(float(gamma))))

    def forward(self, x: torch.Tensor, u_nom: torch.Tensor) -> torch.Tensor:
        gamma = self.log_gamma.exp()              # learn the class-K gain
        gx = self.grad_h_fn(x)
        h = self.h_fn(x)
        a = torch.einsum("bn,bnm->bm", gx, self.g_fn(x))      # Lg h
        b = (gx * self.f_fn(x)).sum(-1) + gamma * h           # Lf h + gamma h
        viol = torch.relu(-((a * u_nom).sum(-1) + b))
        return u_nom + a * (viol / (a.pow(2).sum(-1) + 1e-9)).unsqueeze(-1)


# ---------------------------------------------------------------------------
# 4. Using a filter during RL training
# ---------------------------------------------------------------------------


class SafetyFilterWrapper:
    """Gym-style wrapper that filters every action the agent proposes.

    Three design decisions matter far more than the wrapper itself:

    ``credit``
        Which action does the critic learn about?  Storing ``u_nom`` while the
        environment executed ``u_safe`` creates an off-by-one-policy mismatch.
        Store the *executed* action (and, if the algorithm allows, add a
        penalty on the correction) -- this is the choice made in recent
        safety-filtered actor-critic work.

    ``shaping``
        Penalising ``||u_safe - u_nom||`` teaches the policy the constraint
        instead of leaving it to lean on the filter, which is the core idea of
        CBF-RL (Yang, Werner, de Sa & Ames, 2025/26): enforce the CBF *in
        training* so the deployed policy needs no runtime filter.

    ``deployment``
        Keep the filter at deployment unless you have evidence the policy
        internalised the constraint.  Two safety nets are better than one.
    """

    def __init__(self, env, filter_fn, penalty: float = 0.0,
                 record: bool = True):
        self.env, self.filter_fn, self.penalty = env, filter_fn, penalty
        self.record = record
        self.stats = {"interventions": 0, "steps": 0, "correction": 0.0}

    def reset(self, **kw):
        return self.env.reset(**kw)

    def step(self, action):
        x = self.env.unwrapped.state                  # or an observer estimate
        u_safe = self.filter_fn(x, np.asarray(action, dtype=float))
        dev = float(np.linalg.norm(u_safe - action))
        obs, r, term, trunc, info = self.env.step(u_safe)
        if self.record:
            self.stats["steps"] += 1
            self.stats["correction"] += dev
            self.stats["interventions"] += int(dev > 1e-6)
        info["u_executed"] = u_safe                   # store THIS in the buffer
        info["cbf_correction"] = dev
        return obs, r - self.penalty * dev ** 2, term, trunc, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def cbf_penalty_reward(h_value: float, h_dot_value: float, gamma: float,
                       weight: float = 1.0) -> float:
    """Reward shaping that rewards *satisfying the CBF inequality itself*.

        r_cbf = -weight * max(0, -(h_dot + gamma*h))

    Softer than a filter and differentiable through the policy, but it buys no
    guarantee on its own: use it to reduce how often a hard filter has to
    intervene, never as the only safety mechanism.
    """
    return -weight * max(0.0, -(h_dot_value + gamma * h_value))


# ---------------------------------------------------------------------------
# 5. Sanity checklist before trusting a learned barrier
# ---------------------------------------------------------------------------

CHECKLIST = """
[ ] Descent condition verified over the whole operating domain, not samples
[ ] Input limits included in the descent check (sup over U, not over R^m)
[ ] Learning error bounded and folded into the condition (robust / ISSf term)
[ ] Relative degree handled (HOCBF chain or backup-set construction)
[ ] Sampled-data margin added for the real control period
[ ] State estimate error accounted for (MR-CBF term) if h uses perception
[ ] Deadlock / undesired equilibria checked in the intended workspace
[ ] Behaviour logged at deployment: intervention rate is your early warning
"""
