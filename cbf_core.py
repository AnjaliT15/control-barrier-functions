"""
cbf_core.py -- A minimal, dependency-light Control Barrier Function toolbox.

Only numpy + scipy are required.  Everything is written to be read rather than
to be fast: each function maps one-to-one onto an equation in the report
`control_barrier_functions_report.html`.

Contents
--------
    solve_qp            : tiny dense QP solver (SLSQP) for safety filters
    solve_socp_like     : filter with a second-order-cone term (MR-CBF / robust CBF)
    cbf_filter          : generic first-order CBF-QP safety filter
    hocbf_constraint    : High-Order CBF (HOCBF) row builder for relative degree 2
    ClassK              : the extended class-K functions used in the CBF condition
    simulate            : fixed-step RK4 / Euler closed-loop simulation with ZOH control

Author: prepared for a technical report on control barrier functions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# 1. Extended class-K functions  alpha : R -> R, alpha(0)=0, strictly increasing
# ----------------------------------------------------------------------------


class ClassK:
    """Extended class-K_infinity functions used in the CBF inequality.

    The CBF condition is   sup_u [ Lf h(x) + Lg h(x) u ]  >=  -alpha( h(x) ).

    * ``linear(gamma)``  -> alpha(s) = gamma * s          (the usual choice)
    * ``cubic(gamma)``   -> alpha(s) = gamma * s**3       (soft near the boundary)
    * ``sqrt_signed(g)`` -> alpha(s) = g*sign(s)*sqrt|s|  (aggressive near boundary)
    """

    @staticmethod
    def linear(gamma: float):
        f = lambda s: gamma * s
        f.deriv = lambda s: gamma * np.ones_like(np.asarray(s, dtype=float))
        f.label = rf"$\alpha(s)={gamma:g}s$"
        return f

    @staticmethod
    def cubic(gamma: float):
        f = lambda s: gamma * s ** 3
        f.deriv = lambda s: 3.0 * gamma * np.asarray(s, dtype=float) ** 2
        f.label = rf"$\alpha(s)={gamma:g}s^3$"
        return f

    @staticmethod
    def sqrt_signed(gamma: float):
        f = lambda s: gamma * np.sign(s) * np.sqrt(np.abs(s))
        f.deriv = lambda s: 0.5 * gamma / np.sqrt(np.abs(s) + 1e-9)
        f.label = rf"$\alpha(s)={gamma:g}\,\mathrm{{sgn}}(s)\sqrt{{|s|}}$"
        return f


# ----------------------------------------------------------------------------
# 2. A very small dense QP solver:  min 1/2 u' H u + f' u  s.t.  A u <= b, lb <= u <= ub
# ----------------------------------------------------------------------------


def solve_qp(H, f, A=None, b=None, lb=None, ub=None, u0=None):
    """Solve a small strictly-convex QP with SLSQP.

    Returns ``(u, info)`` where ``info['feasible']`` is False when SLSQP could
    not satisfy the inequality constraints (this happens for CBF-QPs with tight
    input bounds -- exactly the pointwise-feasibility problem discussed in the
    report).
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    f = np.atleast_1d(np.asarray(f, dtype=float))
    m = f.size
    if u0 is None:
        u0 = np.zeros(m)

    cost = lambda u: 0.5 * u @ H @ u + f @ u
    grad = lambda u: H @ u + f

    cons = []
    if A is not None and np.size(A) > 0:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        cons.append({"type": "ineq", "fun": lambda u: b - A @ u, "jac": lambda u: -A})

    bounds = None
    if lb is not None or ub is not None:
        lb = -np.inf * np.ones(m) if lb is None else np.atleast_1d(lb) * np.ones(m)
        ub = +np.inf * np.ones(m) if ub is None else np.atleast_1d(ub) * np.ones(m)
        bounds = list(zip(lb, ub))

    res = minimize(cost, u0, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 80, "ftol": 1e-9})
    u = res.x
    viol = 0.0
    if A is not None and np.size(A) > 0:
        viol = float(np.max(A @ u - b))
    return u, {"feasible": viol <= 1e-6, "violation": viol, "status": res.status}


def solve_socp_like(u_nom, a_vec, c_scalar, rho, lb=None, ub=None):
    """Filter with a second-order-cone safety term (MR-CBF / robust CBF form).

    Solves       min  ||u - u_nom||^2
                 s.t. a_vec . u + c_scalar - rho * ||u|| >= 0
                      lb <= u <= ub

    The ``rho*||u||`` term is what makes measurement-robust and
    actuation-uncertainty-robust CBF conditions conic rather than affine.
    """
    m = np.size(u_nom)
    cost = lambda u: float(np.sum((u - u_nom) ** 2))
    grad = lambda u: 2.0 * (u - u_nom)
    con = {
        "type": "ineq",
        "fun": lambda u: a_vec @ u + c_scalar - rho * np.linalg.norm(u),
        "jac": lambda u: a_vec - rho * u / (np.linalg.norm(u) + 1e-9),
    }
    bounds = None
    if lb is not None or ub is not None:
        lb = -np.inf * np.ones(m) if lb is None else np.atleast_1d(lb) * np.ones(m)
        ub = +np.inf * np.ones(m) if ub is None else np.atleast_1d(ub) * np.ones(m)
        bounds = list(zip(lb, ub))
    res = minimize(cost, np.asarray(u_nom, dtype=float), jac=grad, bounds=bounds,
                   constraints=[con], method="SLSQP",
                   options={"maxiter": 120, "ftol": 1e-10})
    u = res.x
    slack = float(a_vec @ u + c_scalar - rho * np.linalg.norm(u))
    return u, {"feasible": slack >= -1e-6, "slack": slack}


# ----------------------------------------------------------------------------
# 3. The generic first-order CBF-QP safety filter
# ----------------------------------------------------------------------------


def cbf_filter(u_nom, Lfh, Lgh, h, alpha, lb=None, ub=None, weight=None,
               slack_penalty=None):
    """Minimally invasive safety filter (the "CBF-QP").

        u* = argmin_u ||u - u_nom||_W^2
             s.t.  Lfh + Lgh . u >= -alpha(h)                 (CBF condition)
                   lb <= u <= ub                              (actuator limits)

    ``Lfh``, ``h`` are scalars; ``Lgh`` is a row of length m.  Several safety
    constraints are supported by stacking: pass ``Lfh``, ``h`` as 1-D arrays and
    ``Lgh`` as a (k, m) matrix.

    ``slack_penalty`` (optional) turns the hard constraint into a soft one,
    ``... >= -alpha(h) - delta`` with cost ``slack_penalty * delta^2``; this is the
    standard relaxation used when pointwise feasibility cannot be guaranteed.
    """
    u_nom = np.atleast_1d(np.asarray(u_nom, dtype=float))
    m = u_nom.size
    Lfh = np.atleast_1d(np.asarray(Lfh, dtype=float))
    h = np.atleast_1d(np.asarray(h, dtype=float))
    Lgh = np.atleast_2d(np.asarray(Lgh, dtype=float))
    if Lgh.shape[0] != Lfh.size:          # allow (m,) input for a single constraint
        Lgh = Lgh.reshape(Lfh.size, m)

    # A u <= b  form:  -Lgh u <= Lfh + alpha(h)
    A = -Lgh
    b = Lfh + alpha(h)

    W = np.eye(m) if weight is None else np.asarray(weight, dtype=float)

    if slack_penalty is None:
        u, info = solve_qp(2 * W, -2 * W @ u_nom, A, b, lb, ub, u0=u_nom)
        info["slack"] = 0.0
        return u, info

    # augmented decision variable z = [u; delta], delta >= 0
    k = Lfh.size
    H = np.zeros((m + 1, m + 1))
    H[:m, :m] = 2 * W
    H[m, m] = 2 * slack_penalty
    f = np.concatenate([-2 * W @ u_nom, [0.0]])
    Aa = np.hstack([A, -np.ones((k, 1))])
    lba = np.concatenate([-np.inf * np.ones(m) if lb is None else np.atleast_1d(lb) * np.ones(m), [0.0]])
    uba = np.concatenate([+np.inf * np.ones(m) if ub is None else np.atleast_1d(ub) * np.ones(m), [np.inf]])
    z, info = solve_qp(H, f, Aa, b, lba, uba, u0=np.concatenate([u_nom, [0.0]]))
    info["slack"] = float(z[m])
    return z[:m], info


# ----------------------------------------------------------------------------
# 4. High-Order CBF (relative degree 2) constraint row
# ----------------------------------------------------------------------------


def hocbf_rd2(h0, dh0_dt, Lf2h, LgLfh, alpha1, alpha2, dalpha1=None):
    """Build the HOCBF constraint row for a relative-degree-2 constraint.

        psi0(x) = h0(x)
        psi1(x) = psi0_dot + alpha1(psi0)              (= dh0_dt + alpha1(h0))
        require   psi1_dot + alpha2(psi1) >= 0
                  Lf^2 h + Lg Lf h . u + d/dt[alpha1(h0)] + alpha2(psi1) >= 0

    With alpha1 linear (gain k1) the extra term is simply ``k1 * dh0_dt``.
    Returns ``(Lgh_row, rhs)`` so that the CBF condition reads
    ``Lgh_row . u >= -rhs``.
    """
    psi1 = dh0_dt + alpha1(h0)
    if dalpha1 is None:
        dalpha1 = getattr(alpha1, "deriv", lambda s: 0.0)
    rhs = Lf2h + dalpha1(h0) * dh0_dt + alpha2(psi1)
    return np.atleast_1d(LgLfh), float(rhs), float(psi1)


# ----------------------------------------------------------------------------
# 5. Simulation helpers (zero-order-hold control, as on real hardware)
# ----------------------------------------------------------------------------


def simulate(f_dyn, controller, x0, dt, T, method="rk4", substeps=1):
    """Closed-loop simulation with zero-order-hold control.

    ``f_dyn(t, x, u) -> xdot`` and ``controller(t, x) -> (u, aux_dict)``.
    The control is computed once per ``dt`` and held (as on real hardware);
    ``substeps`` refines the *integration* only, which lets us separate
    integration error from sampled-data (ZOH) effects.
    """
    n_steps = int(round(T / dt))
    xs = np.zeros((n_steps + 1, np.size(x0)))
    us, ts, auxs = [], [], []
    xs[0] = np.asarray(x0, dtype=float)
    t = 0.0
    for k in range(n_steps):
        u, aux = controller(t, xs[k])
        x = xs[k].copy()
        hh = dt / substeps
        for _ in range(substeps):
            if method == "euler":
                x = x + hh * f_dyn(t, x, u)
            else:  # rk4
                k1 = f_dyn(t, x, u)
                k2 = f_dyn(t + hh / 2, x + hh / 2 * k1, u)
                k3 = f_dyn(t + hh / 2, x + hh / 2 * k2, u)
                k4 = f_dyn(t + hh, x + hh * k3, u)
                x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += hh
        xs[k + 1] = x
        us.append(np.atleast_1d(u))
        ts.append(t - dt)
        auxs.append(aux)
    return {
        "t": np.array(ts + [t]),
        "x": xs,
        "u": np.array(us),
        "aux": auxs,
    }


# ----------------------------------------------------------------------------
# 6. Barrier functions used throughout the report
# ----------------------------------------------------------------------------


def h_disc(p, centre, radius):
    """Signed 'squared distance' barrier for a circular obstacle.

    h(p) = ||p - c||^2 - r^2  >= 0  outside the disc.  Squared form keeps h
    smooth everywhere (the plain distance is non-smooth at the centre).
    """
    d = np.asarray(p, dtype=float) - np.asarray(centre, dtype=float)
    return float(d @ d - radius ** 2)


def grad_h_disc(p, centre, radius=None):
    d = np.asarray(p, dtype=float) - np.asarray(centre, dtype=float)
    return 2.0 * d


def h_dist(p, centre, radius):
    """Clearance barrier  h(p) = ||p - c|| - r  (metres of clearance).

    Smooth everywhere except at the obstacle centre; its gradient is a unit
    vector, which makes the CBF condition easy to read: ``n(p) . u >= -alpha(h)``
    limits the approach speed along the line of sight.
    """
    d = np.asarray(p, dtype=float) - np.asarray(centre, dtype=float)
    return float(np.linalg.norm(d) - radius)


def grad_h_dist(p, centre, radius=None):
    d = np.asarray(p, dtype=float) - np.asarray(centre, dtype=float)
    return d / (np.linalg.norm(d) + 1e-12)


def h_halfspace(p, normal, offset):
    """h(p) = normal . p - offset >= 0."""
    return float(np.asarray(normal, dtype=float) @ np.asarray(p, dtype=float) - offset)


def soft_min(h_list, kappa=10.0):
    """Log-sum-exp soft minimum: a single smooth barrier from many constraints.

        h_soft = -(1/kappa) log sum_i exp(-kappa h_i)   <=  min_i h_i

    This is the aggregation trick used to collapse many CBF constraints into
    one (and hence to admit a closed-form filter).
    """
    h = np.asarray(h_list, dtype=float)
    return float(-np.log(np.sum(np.exp(-kappa * h))) / kappa)


def soft_min_weights(h_list, kappa=10.0):
    """Gradient weights of ``soft_min``: dh_soft/dh_i."""
    h = np.asarray(h_list, dtype=float)
    w = np.exp(-kappa * (h - np.min(h)))
    return w / np.sum(w)
