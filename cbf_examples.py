"""
cbf_examples.py -- every numerical experiment and figure in the CBF report.

Usage
-----
    python cbf_examples.py             # regenerate every figure into ./figs
    python cbf_examples.py fig05 fig07 # regenerate selected figures

Each ``figNN`` function is self-contained and heavily commented: it is meant to
be read alongside the corresponding section of the report and edited freely.
Requires only numpy, scipy, matplotlib (plus scikit-learn for the two
data-driven figures, Sections 8.2 and 8.3).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow

from cbf_core import (ClassK, cbf_filter, solve_qp, solve_socp_like, simulate,
                      h_disc, grad_h_disc, h_dist, grad_h_dist, hocbf_rd2,
                      soft_min, soft_min_weights)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGDIR, exist_ok=True)

# ----------------------------------------------------------------------------
# House style -- a cool "blueprint" palette shared by report and figures
# ----------------------------------------------------------------------------
INK = "#17222B"
PAPER = "#F7F9FA"
RULE = "#B9C5CC"
SAFE = "#1F6F5C"      # safe set / satisfied constraint
UNSAFE = "#A43326"    # unsafe set / violation
NOMINAL = "#8C99A3"   # unfiltered nominal controller
FILTERED = "#2C5F8E"  # CBF-filtered controller
ACCENT = "#B07A1E"    # third signal / margin

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.linewidth": 0.45,
    "grid.alpha": 0.7,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "lines.linewidth": 1.8,
})


def _save(fig, name):
    path = os.path.join(FIGDIR, name + ".png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", path)
    return path


def _tidy(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=11, pad=7)
    return ax


# ============================================================================
# FIG 01 -- Safety as forward invariance: Nagumo's condition on the boundary
# ============================================================================

def fig01():
    """Geometry of safety: the safe set C = {h >= 0}, its boundary, and the
    requirement that the closed-loop vector field never point outward."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # ---- left: the sets and Nagumo's condition -----------------------------
    ax = axes[0]
    xs = np.linspace(-2.4, 2.4, 400)
    ys = np.linspace(-2.0, 2.0, 400)
    X, Y = np.meshgrid(xs, ys)
    # a deliberately non-circular safe set so the picture is not trivial
    H = 1.0 - (X ** 2) / 2.4 - (Y ** 2) / 1.1 + 0.35 * np.sin(2.2 * X) * Y

    ax.contourf(X, Y, H, levels=[0, 1e9], colors=[SAFE], alpha=0.13)
    ax.contourf(X, Y, H, levels=[-1e9, 0], colors=[UNSAFE], alpha=0.09)
    cs = ax.contour(X, Y, H, levels=[0], colors=[SAFE], linewidths=2.0)
    for lev, lw in [(0.25, 0.6), (0.5, 0.6), (0.75, 0.6)]:
        ax.contour(X, Y, H, levels=[lev], colors=[SAFE], linewidths=lw, alpha=0.5)

    # normals and admissible field directions at three boundary points
    def h_fun(p):
        x, y = p
        return 1.0 - x ** 2 / 2.4 - y ** 2 / 1.1 + 0.35 * np.sin(2.2 * x) * y

    def grad_h(p, eps=1e-5):
        x, y = p
        gx = (h_fun((x + eps, y)) - h_fun((x - eps, y))) / (2 * eps)
        gy = (h_fun((x, y + eps)) - h_fun((x, y - eps))) / (2 * eps)
        return np.array([gx, gy])

    seg = cs.allsegs[0][0]
    for idx, sign in zip([int(0.12 * len(seg)), int(0.47 * len(seg)), int(0.80 * len(seg))],
                         [+1, +1, -1]):
        p = seg[idx]
        g = grad_h(p)
        g = g / np.linalg.norm(g)
        # inward normal (grad h points toward increasing h, i.e. inward)
        ax.arrow(p[0], p[1], 0.45 * g[0], 0.45 * g[1], width=0.012,
                 head_width=0.09, color=SAFE, length_includes_head=True, zorder=5)
        tang = np.array([-g[1], g[0]])
        vec = 0.42 * (tang * 0.9 + sign * g * 0.55)
        ax.arrow(p[0], p[1], vec[0], vec[1], width=0.012, head_width=0.09,
                 color=FILTERED if sign > 0 else UNSAFE,
                 length_includes_head=True, zorder=6)

    ax.text(0.05, 0.72, r"$\mathcal{C}=\{x: h(x)\geq 0\}$", color=SAFE,
            transform=ax.transAxes, fontsize=11)
    ax.text(0.05, 0.05, r"$h(x)<0$", color=UNSAFE, transform=ax.transAxes, fontsize=11)
    ax.text(0.50, 0.94, r"$\nabla h\cdot \dot{x}\geq 0$  admissible", color=FILTERED,
            transform=ax.transAxes, fontsize=9.5)
    ax.text(0.50, 0.87, r"$\nabla h\cdot \dot{x}< 0$  leaves $\mathcal{C}$", color=UNSAFE,
            transform=ax.transAxes, fontsize=9.5)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.grid(False)
    _tidy(ax, r"$x_1$", r"$x_2$", "Safety is forward invariance of a set")

    # ---- right: what the CBF inequality does to h along a trajectory -------
    ax = axes[1]
    t = np.linspace(0, 6, 600)
    h0 = 1.0
    for gamma, style in [(0.4, "-"), (1.2, "--"), (3.0, ":")]:
        ax.plot(t, h0 * np.exp(-gamma * t), style, color=SAFE,
                label=rf"$h(0)e^{{-\gamma t}}$, $\gamma={gamma}$")
    # an actual trajectory that respects the bound but is not the bound
    hh = h0 * np.exp(-0.4 * t) * (1.0 + 0.55 * (1.0 - np.cos(2.6 * t)))
    ax.plot(t, hh, color=FILTERED, lw=2.2,
            label=r"$h(x(t))$ under $\dot h\geq-0.4\,h$")
    ax.axhline(0, color=UNSAFE, lw=1.4)
    ax.fill_between(t, -0.4, 0, color=UNSAFE, alpha=0.09)
    ax.set_ylim(-0.4, 2.25)
    ax.text(4.2, -0.26, "unsafe", color=UNSAFE, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=9)
    _tidy(ax, r"time $t$", r"$h(x(t))$",
          r"With $\dot h \geq -\gamma h$, $h$ may decay but never crosses zero")
    fig.tight_layout()
    return _save(fig, "fig01_forward_invariance")


# ============================================================================
# FIG 02 -- Choice of class-K function: conservatism vs. aggressiveness
# ============================================================================

def fig02():
    """Same plant, same barrier, three class-K functions.  The class-K choice
    is the single most consequential 'free parameter' of a CBF design."""
    # single-integrator robot, disc obstacle, nominal = go-to-goal
    centre, radius = np.array([0.0, 0.0]), 1.0
    goal = np.array([3.2, 0.25])
    x0 = np.array([-3.2, -0.25])
    u_max = 1.2

    def run(alpha):
        def controller(t, p):
            e = goal - p
            u_nom = np.clip(1.0 * e, -u_max, u_max)
            h = h_dist(p, centre, radius)
            g = grad_h_dist(p, centre)
            u, info = cbf_filter(u_nom, 0.0, g, h, alpha, lb=-u_max, ub=u_max)
            return u, {"h": h, "u_nom": u_nom, "dev": np.linalg.norm(u - u_nom)}
        sim = simulate(lambda t, p, u: u, controller, x0, 0.02, 12.0)
        return sim

    alphas = [ClassK.linear(0.35), ClassK.linear(6.0), ClassK.cubic(6.0)]
    labels = [r"$\alpha(s)=0.35\,s$ — cautious, wide detour",
              r"$\alpha(s)=6\,s$ — aggressive, hugs $\partial\mathcal{C}$",
              r"$\alpha(s)=6\,s^3$ — soft far out, firm at the boundary"]
    colors = [FILTERED, ACCENT, SAFE]

    fig = plt.figure(figsize=(9.6, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.45, wspace=0.28)
    axp = fig.add_subplot(gs[:, 0])
    axh = fig.add_subplot(gs[0, 1])
    axd = fig.add_subplot(gs[1, 1])

    axp.add_patch(Circle(centre, radius, facecolor=UNSAFE, alpha=0.15,
                         edgecolor=UNSAFE, lw=1.6))
    axp.text(centre[0], centre[1], r"$h<0$", ha="center", va="center", color=UNSAFE)
    for alpha, lab, c in zip(alphas, labels, colors):
        sim = run(alpha)
        axp.plot(sim["x"][:, 0], sim["x"][:, 1], color=c, label=lab)
        h = np.array([a["h"] for a in sim["aux"]])
        dev = np.array([a["dev"] for a in sim["aux"]])
        axh.plot(sim["t"][:-1], h, color=c)
        axd.plot(sim["t"][:-1], dev, color=c)
    axp.plot(*x0, "o", color=INK, ms=5)
    axp.plot(*goal, "*", color=INK, ms=11)
    axp.annotate("start", x0 + np.array([0.1, -0.45]), color=INK, fontsize=9)
    axp.annotate("goal", goal + np.array([-0.35, 0.35]), color=INK, fontsize=9)
    axp.set_aspect("equal")
    axp.set_xlim(-3.6, 3.6)
    axp.set_ylim(-2.3, 2.3)
    axp.legend(loc="lower center", fontsize=8.4)
    _tidy(axp, r"$p_1$", r"$p_2$", "Trajectories under one barrier, three class-$\\mathcal{K}$ choices")

    axh.axhline(0, color=UNSAFE, lw=1.2)
    _tidy(axh, None, r"$h(p)$ [m]", "Clearance barrier")
    axh.set_ylim(-0.15, 2.6)
    _tidy(axd, "time [s]", r"$\|u-u_{\rm nom}\|$", "Filter intervention")
    return _save(fig, "fig02_class_k")


# ============================================================================
# FIG 03 -- The CBF-QP as a minimally invasive safety filter
# ============================================================================

def fig03():
    """Anatomy of the safety filter: nominal vs filtered input, the admissible
    input set K_cbf(x) at a snapshot, and the barrier value over time."""
    centre, radius = np.array([0.0, 0.0]), 1.0
    goal = np.array([3.0, 1.2])
    x0 = np.array([-3.0, -1.0])
    u_max = 1.0
    alpha = ClassK.linear(0.55)

    log = []

    def u_go_to_goal(p):
        """Constant-speed go-to-goal: the nominal task controller."""
        e = goal - p
        n = np.linalg.norm(e)
        return u_max * e / n if n > 0.08 else np.zeros(2)

    def controller(t, p):
        u_nom = u_go_to_goal(p)
        h = h_dist(p, centre, radius)
        g = grad_h_dist(p, centre)
        u, info = cbf_filter(u_nom, 0.0, g, h, alpha, lb=-u_max, ub=u_max)
        log.append((t, p.copy(), u_nom.copy(), u.copy(), h, g.copy()))
        return u, {"h": h, "u_nom": u_nom}

    sim_f = simulate(lambda t, p, u: u, controller, x0, 0.02, 10.0)

    def nominal_only(t, p):
        return u_go_to_goal(p), {}
    sim_n = simulate(lambda t, p, u: u, nominal_only, x0, 0.02, 10.0)

    fig = plt.figure(figsize=(10.4, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], hspace=0.55, wspace=0.38)
    axp = fig.add_subplot(gs[:, 0])
    axk = fig.add_subplot(gs[:, 1])
    axu = fig.add_subplot(gs[0, 2])
    axh = fig.add_subplot(gs[1, 2])

    axp.add_patch(Circle(centre, radius, facecolor=UNSAFE, alpha=0.15,
                         edgecolor=UNSAFE, lw=1.6))
    axp.plot(sim_n["x"][:, 0], sim_n["x"][:, 1], color=NOMINAL, ls="--",
             label="nominal only")
    axp.plot(sim_f["x"][:, 0], sim_f["x"][:, 1], color=FILTERED,
             label="CBF-QP filtered")
    axp.plot(*goal, "*", color=INK, ms=11)
    axp.plot(*x0, "o", color=INK, ms=4.5)
    axp.set_aspect("equal")
    axp.set_xlim(-3.4, 3.4)
    axp.set_ylim(-2.2, 2.2)
    axp.legend(loc="lower right", fontsize=8.6)
    _tidy(axp, r"$p_1$", r"$p_2$", "Closed loop")

    # snapshot at the instant of largest intervention
    devs = [np.linalg.norm(s[3] - s[2]) for s in log]
    k_snap = int(np.argmax(devs))
    t_s, p_s, unom_s, u_s, h_s, g_s = log[k_snap]
    axp.plot([p_s[0]], [p_s[1]], "o", color=FILTERED, ms=6, zorder=6)
    axp.annotate(rf"$t={t_s:.1f}$ s", p_s + np.array([0.12, -0.55]), fontsize=8.5,
                 color=FILTERED)

    U1, U2 = np.meshgrid(np.linspace(-1.15, 1.15, 320), np.linspace(-1.15, 1.15, 320))
    val = g_s[0] * U1 + g_s[1] * U2 + alpha(h_s)
    axk.contourf(U1, U2, (val >= 0).astype(float), levels=[0.5, 2],
                 colors=[SAFE], alpha=0.16)
    axk.contour(U1, U2, val, levels=[0], colors=[SAFE], linewidths=1.6)
    axk.add_patch(Rectangle((-u_max, -u_max), 2 * u_max, 2 * u_max, fill=False,
                            edgecolor=INK, lw=1.0, ls=":"))
    axk.plot([unom_s[0]], [unom_s[1]], "o", color=NOMINAL, ms=7)
    axk.plot([u_s[0]], [u_s[1]], "o", color=FILTERED, ms=7)
    axk.annotate("", xy=u_s, xytext=unom_s,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    axk.text(unom_s[0] - 0.10, unom_s[1] - 0.24, r"$u_{\rm nom}$", color=NOMINAL,
             fontsize=9.5)
    axk.text(u_s[0] - 0.34, u_s[1] + 0.12, r"$u^\star$", color=FILTERED, fontsize=10)
    axk.text(-1.10, 0.95, r"$K_{\rm cbf}(x)$", color=SAFE, fontsize=10)
    axk.text(0.62, -1.06, r"$\mathcal{U}$", color=INK, fontsize=10)
    axk.set_aspect("equal")
    axk.grid(False)
    _tidy(axk, r"$u_1$", r"$u_2$", r"Input space: project onto $K_{\rm cbf}$")

    tt = np.array([s[0] for s in log])
    un = np.array([s[2] for s in log])
    uf = np.array([s[3] for s in log])
    hh = np.array([s[4] for s in log])
    axu.plot(tt, np.linalg.norm(un, axis=1), color=NOMINAL, ls="--",
             label=r"$\|u_{\rm nom}\|$")
    axu.plot(tt, np.linalg.norm(uf, axis=1), color=FILTERED, label=r"$\|u^\star\|$")
    axu.plot(tt, np.array(devs), color=ACCENT, lw=1.3,
             label=r"$\|u^\star-u_{\rm nom}\|$")
    axu.legend(fontsize=7.8, ncol=1, loc="center right")
    _tidy(axu, None, "input magnitude", "Intervention is minimal")

    hn = np.array([h_dist(p, centre, radius) for p in sim_n["x"]])
    axh.plot(sim_n["t"], hn, color=NOMINAL, ls="--")
    axh.plot(tt, hh, color=FILTERED)
    axh.axhline(0, color=UNSAFE, lw=1.2)
    axh.fill_between(sim_n["t"], -1.2, 0, color=UNSAFE, alpha=0.08)
    axh.set_ylim(-1.15, 3.4)
    _tidy(axh, "time [s]", r"$h$ [m]", "Barrier value")
    return _save(fig, "fig03_cbf_qp_filter")


# ============================================================================
# FIG 04 -- Adaptive cruise control: the canonical CLF-CBF-QP benchmark
# ============================================================================

def fig04():
    """Adaptive cruise control (Ames, Grizzle & Tabuada, CDC 2014).

    States  x = (v, z):  ego speed v, headway z to the lead car.
        v_dot = (-F_r(v) + u)/m,      F_r(v) = f0 + f1 v + f2 v^2
        z_dot = v_lead - v
    Safety: keep a 1.8 s time headway,  h(x) = z - 1.8 v >= 0.
    Performance: track a desired cruise speed v_d.
    """
    m, f0, f1, f2 = 1650.0, 0.1, 5.0, 0.25
    v_lead, v_des = 13.89, 24.0
    c_d, g_acc = 0.3, 9.81
    u_lim = c_d * m * g_acc          # +/- 0.3 g wheel force
    tau = 1.8
    alpha = ClassK.linear(1.0)

    Fr = lambda v: f0 + f1 * v + f2 * v ** 2

    def dyn(t, x, u):
        v, z = x
        return np.array([(-Fr(v) + u[0]) / m, v_lead - v])

    def make_controller(use_cbf, alpha=alpha):
        def controller(t, x):
            v, z = x
            # nominal: feedback-linearising cruise controller
            u_nom = np.array([Fr(v) + m * 0.6 * (v_des - v)])
            u_nom = np.clip(u_nom, -u_lim, u_lim)
            h = z - tau * v
            # h_dot = (v_lead - v) + tau*Fr(v)/m - tau*u/m
            Lfh = (v_lead - v) + tau * Fr(v) / m
            Lgh = np.array([-tau / m])
            if use_cbf:
                u, info = cbf_filter(u_nom, Lfh, Lgh, h, alpha, lb=-u_lim, ub=u_lim)
            else:
                u, info = u_nom, {"feasible": True}
            return u, {"h": h, "u_nom": float(u_nom[0]), "feasible": info["feasible"]}
        return controller

    x0 = np.array([20.0, 100.0])
    sim_n = simulate(dyn, make_controller(False), x0, 0.02, 40.0)
    sim_c = simulate(dyn, make_controller(True), x0, 0.02, 40.0)

    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.1))
    ax = axes[0]
    ax.plot(sim_n["t"], sim_n["x"][:, 0], color=NOMINAL, ls="--", label="cruise only")
    ax.plot(sim_c["t"], sim_c["x"][:, 0], color=FILTERED, label="CBF-QP")
    ax.axhline(v_des, color=INK, lw=0.8, ls=":")
    ax.axhline(v_lead, color=SAFE, lw=0.8, ls=":")
    ax.text(26, v_des + 0.4, r"$v_d$", fontsize=9)
    ax.text(26, v_lead + 0.4, r"$v_{\rm lead}$", fontsize=9, color=SAFE)
    ax.legend(fontsize=8.6)
    _tidy(ax, "time [s]", r"$v$ [m/s]", "Ego speed")

    ax = axes[1]
    ax.plot(sim_n["t"], sim_n["x"][:, 1], color=NOMINAL, ls="--")
    ax.plot(sim_c["t"], sim_c["x"][:, 1], color=FILTERED)
    ax.set_ylim(-10, 110)
    ax.axhline(0, color=RULE, lw=0.8)
    _tidy(ax, "time [s]", r"$z$ [m]", "Headway (dashed: collision)")

    ax = axes[2]
    hn = sim_n["x"][:, 1] - tau * sim_n["x"][:, 0]
    hc = sim_c["x"][:, 1] - tau * sim_c["x"][:, 0]
    ax.plot(sim_n["t"], hn, color=NOMINAL, ls="--")
    ax.plot(sim_c["t"], hc, color=FILTERED)
    ax.axhline(0, color=UNSAFE, lw=1.2)
    ax.fill_between(sim_n["t"], -30, 0, color=UNSAFE, alpha=0.08)
    ax.set_ylim(-30, 75)
    _tidy(ax, "time [s]", r"$h=z-1.8v$ [m]", "Safety margin")

    ax = axes[3]
    ax.plot(sim_c["t"][:-1], [a["u_nom"] / (m * g_acc) for a in sim_c["aux"]],
            color=NOMINAL, ls="--", label=r"$u_{\rm nom}$")
    ax.plot(sim_c["t"][:-1], sim_c["u"][:, 0] / (m * g_acc), color=FILTERED,
            label=r"$u^\star$")
    ax.axhline(c_d, color=INK, lw=0.8, ls=":")
    ax.axhline(-c_d, color=INK, lw=0.8, ls=":")
    ax.text(1.5, -c_d + 0.025, r"$\pm 0.3g$ actuator limit", fontsize=8)
    ax.set_ylim(-0.345, 0.345)
    ax.legend(fontsize=8.6, loc="center right")
    _tidy(ax, "time [s]", r"$u/(mg)$", "Wheel force")
    fig.tight_layout()
    return _save(fig, "fig04_acc")


# ============================================================================
# FIG 05 -- Relative degree: why a plain CBF fails and HOCBF/ECBF fixes it
# ============================================================================

def fig05():
    """Double integrator, position constraint -> relative degree 2.

    p_ddot = u with h0(p) = ||p-c||^2 - r^2.  Then Lg h0 = 0: the barrier
    derivative does not contain u at all, so the naive CBF-QP is vacuous.
    The HOCBF chain psi0 = h0, psi1 = h0_dot + a1(h0) restores it.
    """
    centre, radius = np.array([0.0, 0.0]), 1.0
    goal = np.array([3.0, 1.2])
    x0 = np.array([-3.0, -1.0, 1.3, 0.45])    # (p, v)
    u_max = 2.0
    a1, a2 = ClassK.linear(2.5), ClassK.linear(2.5)

    def dyn(t, x, u):
        return np.concatenate([x[2:], u])

    def nominal(p, v):
        """PD go-to-goal; on its own it drives straight through the obstacle."""
        return np.clip(1.1 * (goal - p) - 2.0 * v, -u_max, u_max)

    def ctrl_naive(t, x):
        """Applies the relative-degree-1 condition to h0 -- structurally vacuous:
        Lg h0 = 0, so the QP can never enforce anything through u."""
        p, v = x[:2], x[2:]
        h0 = h_disc(p, centre, radius)
        u, info = cbf_filter(nominal(p, v), float(grad_h_disc(p, centre) @ v),
                             np.zeros(2), h0, ClassK.linear(2.0),
                             lb=-u_max, ub=u_max, slack_penalty=1e4)
        return u, {"h": h0}

    def ctrl_hocbf(t, x):
        p, v = x[:2], x[2:]
        h0 = h_disc(p, centre, radius)
        dh0 = 2.0 * (p - centre) @ v                 # h0_dot
        Lf2h = 2.0 * (v @ v)                         # d/dt of h0_dot, u-free part
        LgLfh = 2.0 * (p - centre)                   # coefficient of u
        row, rhs, psi1 = hocbf_rd2(h0, dh0, Lf2h, LgLfh, a1, a2)
        u, info = cbf_filter(nominal(p, v), rhs, row, 0.0, lambda s: 0.0,
                             lb=-u_max, ub=u_max, slack_penalty=1e5)
        return u, {"h": h0, "psi1": psi1, "slack": info["slack"]}

    sim_a = simulate(dyn, ctrl_naive, x0, 0.01, 9.0)
    sim_b = simulate(dyn, ctrl_hocbf, x0, 0.01, 9.0)

    fig = plt.figure(figsize=(10.0, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.45, wspace=0.3)
    axp = fig.add_subplot(gs[:, 0])
    axh = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, 1])

    axp.add_patch(Circle(centre, radius, facecolor=UNSAFE, alpha=0.15,
                         edgecolor=UNSAFE, lw=1.6))
    axp.plot(sim_a["x"][:, 0], sim_a["x"][:, 1], color=UNSAFE, ls="--",
             label=r"CBF on $h_0$ only ($L_gh_0\equiv 0$)")
    axp.plot(sim_b["x"][:, 0], sim_b["x"][:, 1], color=FILTERED,
             label=r"HOCBF ($\psi_1=\dot h_0+\alpha_1(h_0)$)")
    axp.plot(*goal, "*", color=INK, ms=11)
    axp.set_aspect("equal")
    axp.set_xlim(-3.4, 3.4); axp.set_ylim(-2.1, 2.1)
    axp.legend(loc="lower left", fontsize=8.6)
    _tidy(axp, r"$p_1$", r"$p_2$", "Double integrator, position constraint")

    axh.plot(sim_a["t"][:-1], [a["h"] for a in sim_a["aux"]], color=UNSAFE, ls="--")
    axh.plot(sim_b["t"][:-1], [a["h"] for a in sim_b["aux"]], color=FILTERED)
    axh.axhline(0, color=UNSAFE, lw=1.2)
    axh.fill_between([0, 9], -1.5, 0, color=UNSAFE, alpha=0.08)
    axh.set_ylim(-1.5, 10)
    _tidy(axh, None, r"$h_0=\|p-c\|^2-r^2$", "Barrier value")

    axs.plot(sim_b["t"][:-1], [a["psi1"] for a in sim_b["aux"]], color=SAFE)
    axs.axhline(0, color=RULE, lw=0.9)
    _tidy(axs, "time [s]", r"$\psi_1$", r"HOCBF second-level function stays $\geq 0$")
    return _save(fig, "fig05_hocbf")


# ============================================================================
# FIG 06 -- Model error and disturbance: robust CBF / ISSf-CBF
# ============================================================================

def fig06():
    """Unmodelled drift d(t) with ||d|| <= delta.

        nominal CBF :  grad_h . u                      >= -alpha(h)
        robust CBF  :  grad_h . u - delta*||grad_h||    >= -alpha(h)
        ISSf-CBF    :  grad_h . u                      >= -alpha(h) + ||grad_h||^2/eps

    The robust form assumes a known bound and pays for it up front; the ISSf
    form trades a tunable amount of the safe set for graceful degradation.
    """
    centre, radius = np.array([0.0, 0.0]), 1.0
    goal = np.array([3.0, 1.2])
    x0 = np.array([-3.0, -1.0])
    u_max = 1.0
    alpha = ClassK.linear(0.8)
    eps = 3.0

    state = {"delta": 0.55, "seed": 7}

    def u_go_to_goal(p):
        e = goal - p
        n = np.linalg.norm(e)
        return u_max * e / n if n > 0.08 else np.zeros(2)

    def make_dyn():
        rng = np.random.default_rng(state["seed"])

        def dyn(t, p, u):
            towards = centre - p
            towards = towards / (np.linalg.norm(towards) + 1e-9)
            d = state["delta"] * (0.8 * towards + 0.2 * rng.standard_normal(2))
            return u + d
        return dyn

    def make_ctrl(mode):
        def ctrl(t, p):
            u_nom = u_go_to_goal(p)
            h = h_dist(p, centre, radius)
            g = grad_h_dist(p, centre)
            if mode == "nominal":
                Lfh, a = 0.0, alpha
            elif mode == "robust":
                Lfh, a = -state["delta"] * np.linalg.norm(g), alpha
            else:                                   # ISSf
                Lfh = 0.0
                a = lambda ss: alpha(ss) - float(g @ g) / eps
            u, info = cbf_filter(u_nom, Lfh, g, h, a, lb=-u_max, ub=u_max,
                                 slack_penalty=1e4)
            return u, {"h": h}
        return ctrl

    styles = [("nominal", NOMINAL, "--", r"nominal CBF (ignores $d$)"),
              ("robust", FILTERED, "-", r"robust CBF, $-\delta\|\nabla h\|$"),
              ("issf", SAFE, "-", r"ISSf-CBF, $\epsilon=3$")]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    ax = axes[0]
    ax.add_patch(Circle(centre, radius, facecolor=UNSAFE, alpha=0.15,
                        edgecolor=UNSAFE, lw=1.6))
    traces = {}
    for mode, c, ls, lab in styles:
        state["delta"] = 0.55
        sim = simulate(make_dyn(), make_ctrl(mode), x0, 0.02, 11.0)
        traces[mode] = sim
        ax.plot(sim["x"][:, 0], sim["x"][:, 1], color=c, ls=ls, label=lab)
    ax.plot(*goal, "*", color=INK, ms=11)
    ax.set_aspect("equal"); ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.2, 2.2)
    ax.legend(fontsize=8.2, loc="lower right")
    _tidy(ax, r"$p_1$", r"$p_2$", r"Drift toward the obstacle, $\delta=0.55$")

    ax = axes[1]
    for mode, c, ls, lab in styles:
        sim = traces[mode]
        ax.plot(sim["t"][:-1], [a["h"] for a in sim["aux"]], color=c, ls=ls)
    ax.axhline(0, color=UNSAFE, lw=1.2)
    ax.fill_between([0, 11], -0.8, 0, color=UNSAFE, alpha=0.08)
    ax.set_ylim(-0.8, 2.6)
    _tidy(ax, "time [s]", r"$h$ [m]", r"Robustification restores $h\geq 0$")

    ax = axes[2]
    deltas = np.linspace(0.0, 0.8, 7)
    for mode, c, ls, lab in styles:
        worst = []
        for d in deltas:
            state["delta"] = float(d)
            state["seed"] = 11
            sim = simulate(make_dyn(), make_ctrl(mode), x0, 0.03, 11.0)
            worst.append(min(a["h"] for a in sim["aux"]))
        ax.plot(deltas, worst, ls, color=c, marker="o", ms=3.5)
    ax.axhline(0, color=UNSAFE, lw=1.2)
    ax.fill_between([0, 0.8], -0.8, 0, color=UNSAFE, alpha=0.08)
    ax.set_ylim(-0.8, 1.2)
    _tidy(ax, r"disturbance bound $\delta$", r"$\min_t h$",
          "Worst-case margin vs disturbance size")
    fig.tight_layout()
    return _save(fig, "fig06_robust_issf")


# ============================================================================
# FIG 07 -- Decentralised / graph-structured CBFs for a multi-agent system
# ============================================================================

def fig07():
    """Eight agents swap antipodal positions using only local information.

    Each agent i solves its own QP with one CBF row per *neighbour* inside the
    sensing radius R:

        h_ij(p) = ||p_i - p_j|| - 2 r_s ,
        n_ij . u_i  >=  -1/2 alpha(h_ij)          (responsibility split 50/50)

    The set of active rows changes as agents enter and leave each other's
    sensing radius -- i.e. the constraint structure is a *graph*.  This is the
    classical hand-designed version of what GCBF/GCBF+ learn with a graph
    neural network (Zhang et al., T-RO 2025).
    """
    n_ag, R, r_s, u_max = 8, 1.8, 0.5, 0.8
    alpha = ClassK.linear(1.6)
    swirl = 0.32                                 # rotate the nominal: a common
    Rs = np.array([[np.cos(swirl), -np.sin(swirl)],   # deadlock-breaking trick
                   [np.sin(swirl), np.cos(swirl)]])
    rng = np.random.default_rng(3)
    ang = np.linspace(0, 2 * np.pi, n_ag, endpoint=False)
    P0 = 3.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    P0 += 0.05 * rng.standard_normal(P0.shape)
    rot = 0.18                                   # small swirl breaks symmetry
    Rm = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    G = -(P0 @ Rm.T)

    dt, T = 0.05, 20.0
    P = P0.copy()
    hist, deg_hist, mind_hist, t_hist = [P.copy()], [], [], []
    snap_graph = None
    for k in range(int(T / dt)):
        U = np.zeros_like(P)
        deg = 0
        for i in range(n_ag):
            e = G[i] - P[i]
            ne = np.linalg.norm(e)
            speed = u_max * (0.85 + 0.04 * i)    # heterogeneity breaks symmetry
            u_nom = speed * (Rs @ e) / ne if ne > 0.06 else np.zeros(2)
            rows, Lfh, hs = [], [], []
            for j in range(n_ag):
                if j == i:
                    continue
                d = P[i] - P[j]
                dist = np.linalg.norm(d)
                if dist < R:                       # neighbour -> active row
                    rows.append(d / dist)
                    Lfh.append(0.0)
                    hs.append(dist - 2 * r_s)
                    deg += 1
            if rows:
                u, _ = cbf_filter(u_nom, np.array(Lfh), np.array(rows),
                                  np.array(hs), lambda ss: 0.5 * alpha(ss),
                                  lb=-u_max, ub=u_max, slack_penalty=1e4)
            else:
                u = u_nom
            U[i] = u
        P = P + dt * U
        hist.append(P.copy())
        t_hist.append(k * dt)
        deg_hist.append(deg / n_ag)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        mind_hist.append(D.min())
        if abs(k * dt - 5.0) < dt / 2:
            snap_graph = (P.copy(), D.copy())
    hist = np.array(hist)

    fig = plt.figure(figsize=(11.0, 3.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.0], hspace=0.55,
                          wspace=0.32)
    axt = fig.add_subplot(gs[:, 0])
    axg = fig.add_subplot(gs[:, 1])
    axd = fig.add_subplot(gs[0, 2])
    axk = fig.add_subplot(gs[1, 2])

    cmap = plt.get_cmap("viridis")
    for i in range(n_ag):
        axt.plot(hist[:, i, 0], hist[:, i, 1], color=cmap(i / n_ag), lw=1.4)
        axt.plot(hist[0, i, 0], hist[0, i, 1], "o", color=cmap(i / n_ag), ms=4)
        axt.plot(G[i, 0], G[i, 1], "x", color=cmap(i / n_ag), ms=5)
    axt.set_aspect("equal")
    axt.set_xlim(-3.6, 3.6); axt.set_ylim(-3.6, 3.6)
    _tidy(axt, r"$p_1$", r"$p_2$", "Position swap, 8 agents")

    Ps, D = snap_graph
    for i in range(n_ag):
        for j in range(i + 1, n_ag):
            if D[i, j] < R:
                axg.plot([Ps[i, 0], Ps[j, 0]], [Ps[i, 1], Ps[j, 1]],
                         color=SAFE, lw=1.0, alpha=0.75)
    for i in range(n_ag):
        axg.add_patch(Circle(Ps[i], r_s, facecolor=cmap(i / n_ag), alpha=0.85,
                             edgecolor="none"))
        axg.add_patch(Circle(Ps[i], R, facecolor="none", edgecolor=RULE,
                             lw=0.6, ls=":"))
    axg.set_aspect("equal")
    axg.set_xlim(-3.2, 3.2); axg.set_ylim(-3.2, 3.2)
    axg.grid(False)
    _tidy(axg, r"$p_1$", None, r"Neighbour graph at $t=5$ s ($R=1.8$)")

    axd.plot(t_hist, mind_hist, color=FILTERED)
    axd.axhline(2 * r_s, color=UNSAFE, lw=1.2)
    axd.fill_between(t_hist, 0, 2 * r_s, color=UNSAFE, alpha=0.08)
    axd.set_ylim(0, 6.5)
    axd.text(9.0, 2 * r_s + 0.3, r"$2r_s$", color=UNSAFE, fontsize=8)
    _tidy(axd, None, r"$\min_{ij}\|p_i-p_j\|$", "Closest pair")

    axk.plot(t_hist, deg_hist, color=SAFE)
    _tidy(axk, "time [s]", "mean degree", "Active CBF rows per agent")
    return _save(fig, "fig07_graph_cbf")


# ============================================================================
# FIG 08 -- A known failure mode: undesired equilibria / deadlock
# ============================================================================

def fig08():
    """The CBF-QP is pointwise optimal but says nothing about convergence.

    With the obstacle exactly between start and goal, the minimum-norm safe
    input cancels the goal-seeking term and the closed loop settles on an
    asymptotically stable equilibrium *on the boundary of the safe set*.  A
    small tangential bias in the nominal controller removes it -- which is
    exactly why this is treated as a controller-design problem rather than a
    barrier-design one (Reis et al. 2021; Tan & Dimarogonas 2024; Mestres et
    al. 2026).
    """
    centre, radius = np.array([0.0, 0.0]), 1.0
    goal, x0, u_max = np.array([3.0, 0.0]), np.array([-3.0, 0.0]), 1.0
    alpha = ClassK.linear(1.0)

    def make_ctrl(bias):
        def ctrl(t, p):
            e = goal - p
            ne = np.linalg.norm(e)
            u_nom = u_max * e / ne if ne > 0.05 else np.zeros(2)
            if bias:                                # rotate the nominal slightly
                th = 0.35
                Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                u_nom = Rm @ u_nom
            h, g = h_dist(p, centre, radius), grad_h_dist(p, centre)
            u, _ = cbf_filter(u_nom, 0.0, g, h, alpha, lb=-u_max, ub=u_max)
            return u, {"h": h, "gap": ne, "un": np.linalg.norm(u)}
        return ctrl

    s0 = simulate(lambda t, p, u: u, make_ctrl(False), x0, 0.02, 14.0)
    s1 = simulate(lambda t, p, u: u, make_ctrl(True), x0, 0.02, 14.0)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    ax = axes[0]
    ax.add_patch(Circle(centre, radius, facecolor=UNSAFE, alpha=0.15,
                        edgecolor=UNSAFE, lw=1.6))
    ax.plot(s0["x"][:, 0], s0["x"][:, 1], color=UNSAFE, lw=2.4,
            label="symmetric nominal: deadlock")
    ax.plot(s1["x"][:, 0], s1["x"][:, 1], color=FILTERED,
            label=r"nominal rotated by $20^\circ$")
    ax.plot(-1.0, 0.0, "o", color=UNSAFE, ms=7)
    ax.annotate("stable undesired\nequilibrium", (-1.0, 0.0),
                textcoords="offset points", xytext=(-8, -46), fontsize=8,
                color=UNSAFE, ha="center")
    ax.plot(*goal, "*", color=INK, ms=11)
    ax.set_aspect("equal"); ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.2, 2.2)
    ax.legend(fontsize=8.2, loc="lower right")
    _tidy(ax, r"$p_1$", r"$p_2$", "Safe for ever, never arrives")

    ax = axes[1]
    ax.plot(s0["t"][:-1], [a["gap"] for a in s0["aux"]], color=UNSAFE)
    ax.plot(s1["t"][:-1], [a["gap"] for a in s1["aux"]], color=FILTERED)
    _tidy(ax, "time [s]", r"$\|p-p_{\rm goal}\|$ [m]", "Distance to goal")

    ax = axes[2]
    ax.plot(s0["t"][:-1], [a["un"] for a in s0["aux"]], color=UNSAFE)
    ax.plot(s1["t"][:-1], [a["un"] for a in s1["aux"]], color=FILTERED)
    _tidy(ax, "time [s]", r"$\|u^\star\|$", r"Filtered input collapses to $0$")
    fig.tight_layout()
    return _save(fig, "fig08_deadlock")


# ============================================================================
# FIG 09 -- Sampled-data implementation: the guarantee is continuous-time
# ============================================================================

def fig09():
    """The CBF condition is enforced only at sampling instants.

    Between samples the input is held, so h can dip below zero even though
    every QP was feasible.  Tightening the condition by a margin proportional
    to the step (here  alpha(h) -> alpha(h - L*dt) with L an upper bound on
    |h_dot|) restores the guarantee for the sampled-data loop.
    """
    centre, radius = np.array([0.0, 0.0]), 1.0
    goal, x0, u_max = np.array([3.0, 1.2]), np.array([-3.0, -1.0]), 1.0
    alpha = ClassK.linear(3.0)

    def run(dt, margin):
        def ctrl(t, p):
            e = goal - p
            ne = np.linalg.norm(e)
            u_nom = u_max * e / ne if ne > 0.08 else np.zeros(2)
            h, g = h_dist(p, centre, radius), grad_h_dist(p, centre)
            h_eff = h - (u_max * dt if margin else 0.0)
            u, _ = cbf_filter(u_nom, 0.0, g, h_eff, alpha, lb=-u_max, ub=u_max,
                              slack_penalty=1e4)
            return u, {"h": h}
        # substeps=25 keeps the *integration* exact: the only error is the ZOH
        return simulate(lambda t, p, u: u, ctrl, x0, dt, 12.0, substeps=25)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    for dt, c, ls in [(0.02, SAFE, "-"), (0.20, ACCENT, "-"), (0.50, UNSAFE, "-")]:
        sim = run(dt, margin=False)
        ax.plot(sim["t"][:-1], [a["h"] for a in sim["aux"]], color=c, ls=ls,
                marker="." if dt >= 0.2 else None, ms=3,
                label=rf"$\Delta t={dt}$ s")
    ax.axhline(0, color=UNSAFE, lw=1.2)
    ax.fill_between([0, 12], -0.45, 0, color=UNSAFE, alpha=0.08)
    ax.set_ylim(-0.45, 2.4)
    ax.legend(fontsize=8.4)
    _tidy(ax, "time [s]", r"$h$ [m]", "Zero-order hold erodes the margin")

    ax = axes[1]
    dts = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    for margin, c, lab in [(False, UNSAFE, "plain CBF-QP"),
                           (True, FILTERED, r"tightened by $u_{\max}\Delta t$")]:
        mins = [min(a["h"] for a in run(float(d), margin)["aux"]) for d in dts]
        ax.plot(dts, mins, color=c, marker="o", ms=3.5, label=lab)
    ax.axhline(0, color=UNSAFE, lw=1.2)
    ax.fill_between([0, 0.62], -0.45, 0, color=UNSAFE, alpha=0.08)
    ax.set_ylim(-0.45, 0.55)
    ax.legend(fontsize=8.4)
    _tidy(ax, r"control period $\Delta t$ [s]", r"$\min_t h$",
          "Margin restores sampled-data safety")
    fig.tight_layout()
    return _save(fig, "fig09_sampled_data")


# ============================================================================
# FIG 10 -- Learning the safe set from labelled data (kernel-SVM CBF)
# ============================================================================

def fig10():
    """Data-driven barrier synthesis, step 1: learn the *set*.

    Following the supervised-learning route (Srinivasan, Dabholkar, Coogan &
    Vela, IROS 2020), label sampled states safe / unsafe and fit an RBF-kernel
    SVM.  Its decision function is a smooth candidate barrier whose gradient is
    available in closed form:

        h(x)      = sum_i a_i exp(-g||x-s_i||^2) + b
        grad h(x) = sum_i a_i exp(-g||x-s_i||^2) * (-2g (x - s_i))

    The learned h reproduces a non-convex, non-analytic safe set that would be
    painful to hand-design.  It is a *candidate*: nothing so far enforces the
    CBF descent condition, which is what neural-CBF training and verification
    add (Section 8).
    """
    from sklearn.svm import SVC

    discs = [(np.array([-0.5, 1.0]), 0.9), (np.array([1.4, -0.9]), 0.8)]
    wall = (-2.7, -2.1, -3.0, 0.3)               # x0, x1, y0, y1

    def unsafe(P):
        m = np.zeros(len(P), dtype=bool)
        for c, r in discs:
            m |= np.linalg.norm(P - c, axis=1) < r
        m |= ((P[:, 0] > wall[0]) & (P[:, 0] < wall[1]) &
              (P[:, 1] > wall[2]) & (P[:, 1] < wall[3]))
        return m

    def true_clearance(p):
        d = min(np.linalg.norm(p - c) - r for c, r in discs)
        dx = max(wall[0] - p[0], p[0] - wall[1], 0.0)
        dy = max(wall[2] - p[1], p[1] - wall[3], 0.0)
        inside = (wall[0] < p[0] < wall[1]) and (wall[2] < p[1] < wall[3])
        dw = -min(p[0] - wall[0], wall[1] - p[0], p[1] - wall[2], wall[3] - p[1]) \
            if inside else np.hypot(dx, dy)
        return min(d, dw)

    rng = np.random.default_rng(0)
    N = 4000
    P = np.column_stack([rng.uniform(-4, 4, N), rng.uniform(-3, 3, N)])
    y = np.where(unsafe(P), -1, 1)
    gam = 2.0
    clf = SVC(kernel="rbf", gamma=gam, C=30.0).fit(P, y)
    sv, dc, b = clf.support_vectors_, clf.dual_coef_[0], clf.intercept_[0]

    def h_learned(p):
        k = np.exp(-gam * np.sum((sv - p) ** 2, axis=1))
        return float(dc @ k + b)

    def grad_h_learned(p):
        d = p - sv
        k = np.exp(-gam * np.sum(d ** 2, axis=1))
        return -2.0 * gam * ((dc * k)[:, None] * d).sum(axis=0)

    # ---- closed loop using the *learned* barrier -----------------------------
    goal, x0, u_max = np.array([3.5, -1.0]), np.array([-3.7, 2.3]), 1.0
    alpha = ClassK.linear(1.5)

    def ctrl(t, p):
        e = goal - p
        ne = np.linalg.norm(e)
        u_nom = u_max * e / ne if ne > 0.08 else np.zeros(2)
        h, g = h_learned(p), grad_h_learned(p)
        u, _ = cbf_filter(u_nom, 0.0, g, h, alpha, lb=-u_max, ub=u_max,
                          slack_penalty=1e4)
        return u, {"h": h, "clear": true_clearance(p)}

    sim = simulate(lambda t, p, u: u, ctrl, x0, 0.03, 16.0)

    fig = plt.figure(figsize=(11.4, 3.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.85], hspace=0.55,
                          wspace=0.32)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 2])

    Xg, Yg = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-3, 3, 160))
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    K = np.exp(-gam * ((pts[:, None, :] - sv[None, :, :]) ** 2).sum(-1))
    Hg = (K @ dc + b).reshape(Xg.shape)

    sub = rng.choice(N, 900, replace=False)
    ax1.scatter(P[sub][y[sub] > 0][:, 0], P[sub][y[sub] > 0][:, 1], s=2.0,
                color=SAFE, alpha=0.5)
    ax1.scatter(P[sub][y[sub] < 0][:, 0], P[sub][y[sub] < 0][:, 1], s=2.0,
                color=UNSAFE, alpha=0.6)
    ax1.contour(Xg, Yg, Hg, levels=[0], colors=[INK], linewidths=2.0)
    ax1.set_aspect("equal"); ax1.grid(False)
    _tidy(ax1, r"$x_1$", r"$x_2$", r"Labels and learned $\{h_\theta=0\}$")

    ax2.contourf(Xg, Yg, Hg, levels=16, cmap="BrBG", alpha=0.8)
    ax2.contour(Xg, Yg, Hg, levels=[0], colors=[INK], linewidths=1.8)
    for c, r in discs:
        ax2.add_patch(Circle(c, r, fill=False, edgecolor=UNSAFE, lw=1.3, ls="--"))
    ax2.add_patch(Rectangle((wall[0], wall[2]), wall[1] - wall[0],
                            wall[3] - wall[2], fill=False, edgecolor=UNSAFE,
                            lw=1.3, ls="--"))
    ax2.plot(sim["x"][:, 0], sim["x"][:, 1], color=FILTERED, lw=2.2)
    ax2.plot(*goal, "*", color=INK, ms=11)
    ax2.plot(*x0, "o", color=INK, ms=4.5)
    ax2.set_aspect("equal"); ax2.grid(False)
    _tidy(ax2, r"$x_1$", None, r"Filtering with $h_\theta$ (dashed: truth)")

    ax3.plot(sim["t"][:-1], [a["h"] for a in sim["aux"]], color=FILTERED)
    ax3.axhline(0, color=UNSAFE, lw=1.2)
    _tidy(ax3, None, r"$h_\theta$", "Learned barrier")
    ax4.plot(sim["t"][:-1], [a["clear"] for a in sim["aux"]], color=SAFE)
    ax4.axhline(0, color=UNSAFE, lw=1.2)
    _tidy(ax4, "time [s]", "true clearance [m]", "Ground truth")
    return _save(fig, "fig10_svm_cbf")


# ============================================================================
# FIG 11 -- Learning the *dynamics*: GP residual + robust CBF over episodes
# ============================================================================

def fig11():
    """Data-driven barrier synthesis, step 2: shrink the robust margin.

    The plant has an unknown state-dependent drift d(p).  Episode 0 knows only
    a crude bound, so the robust CBF term -delta*||grad h|| is large and the
    behaviour is very conservative.  After each episode a Gaussian process is
    refitted on the observed residuals and the CBF uses

        grad_h . (u + mu(p)) - beta*||grad h||*sigma(p) >= -alpha(h),

    so the margin collapses as the model improves -- the mechanism behind
    GP-based safe learning and RL-CBF (Cheng et al., AAAI 2019).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    centre, radius = np.array([0.0, 0.0]), 1.0
    goal, x0, u_max = np.array([3.0, 1.2]), np.array([-3.0, -1.0]), 1.0
    alpha = ClassK.linear(1.0)
    beta, prior_sd, d_max, sd_floor = 2.0, 0.22, 0.80, 0.04

    def drift(p):
        """Unknown but bounded state-dependent drift, ||d|| <= 0.35."""
        return 0.18 * np.array([np.sin(1.1 * p[1]) + 0.4,
                                np.cos(0.9 * p[0]) - 0.3])

    def dyn(t, p, u):
        return u + drift(p)

    def fit_gps(X, Y):
        if len(X) < 8:
            return None
        k = ConstantKernel(0.1) * RBF(length_scale=1.4) + WhiteKernel(1e-4)
        return [GaussianProcessRegressor(kernel=k, alpha=1e-6,
                                         normalize_y=True).fit(X, Y[:, j])
                for j in (0, 1)]

    def run_episode(gps):
        rec = {"X": [], "Y": [], "sig": []}

        def ctrl(t, p):
            e = goal - p
            ne = np.linalg.norm(e)
            u_nom = u_max * e / ne if ne > 0.08 else np.zeros(2)
            h, g = h_dist(p, centre, radius), grad_h_dist(p, centre)
            if gps is None:
                mu, sd = np.zeros(2), prior_sd * np.ones(2)
                margin = d_max * np.linalg.norm(g)
            else:
                pr = [m.predict(p[None, :], return_std=True) for m in gps]
                mu = np.array([pr[0][0][0], pr[1][0][0]])
                sd = np.maximum(np.array([pr[0][1][0], pr[1][1][0]]), sd_floor)
                margin = beta * np.linalg.norm(g) * float(np.linalg.norm(sd))
            Lfh = float(g @ mu) - margin
            u, _ = cbf_filter(u_nom, Lfh, g, h, alpha, lb=-u_max, ub=u_max,
                              slack_penalty=1e4)
            rec["X"].append(p.copy())
            rec["Y"].append(drift(p))            # measured residual
            rec["sig"].append(float(np.linalg.norm(sd)))
            return u, {"h": h, "margin": margin}
        sim = simulate(dyn, ctrl, x0, 0.05, 14.0)
        return sim, rec

    X, Y, gps = [], [], None
    eps, sims = [], []
    for ep in range(5):
        sim, rec = run_episode(gps)
        sims.append(sim)
        eps.append({
            "min_h": min(a["h"] for a in sim["aux"]),
            "margin": float(np.mean([a["margin"] for a in sim["aux"]])),
            "sigma": float(np.mean(rec["sig"])),
            "len": float(np.sum(np.linalg.norm(np.diff(sim["x"], axis=0), axis=1))),
        })
        X += rec["X"][::6]
        Y += rec["Y"][::6]
        gps = fit_gps(np.array(X), np.array(Y))

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    cmap = plt.get_cmap("cividis")
    ax = axes[0]
    ax.add_patch(Circle(centre, radius, facecolor=UNSAFE, alpha=0.15,
                        edgecolor=UNSAFE, lw=1.6))
    for i, sim in enumerate(sims):
        ax.plot(sim["x"][:, 0], sim["x"][:, 1], color=cmap(i / 4.6),
                label=f"episode {i}")
    ax.plot(*goal, "*", color=INK, ms=11)
    ax.set_aspect("equal"); ax.set_xlim(-3.4, 3.6); ax.set_ylim(-2.4, 2.4)
    ax.legend(fontsize=7.8, loc="lower right")
    _tidy(ax, r"$p_1$", r"$p_2$", "Conservatism falls as the GP learns")

    ax = axes[1]
    for i, sim in enumerate(sims):
        ax.plot(sim["t"][:-1], [a["h"] for a in sim["aux"]], color=cmap(i / 4.6))
    ax.axhline(0, color=UNSAFE, lw=1.2)
    ax.fill_between([0, 14], -0.4, 0, color=UNSAFE, alpha=0.08)
    ax.set_ylim(-0.4, 2.6)
    _tidy(ax, "time [s]", r"$h$ [m]", r"Barrier value per episode")

    ax = axes[2]
    k = np.arange(5)
    ax.plot(k, [e["margin"] for e in eps], color=FILTERED, marker="o", ms=4,
            label=r"mean robust margin")
    ax.plot(k, [e["min_h"] for e in eps], color=SAFE, marker="s", ms=4,
            label=r"$\min_t h$")
    ax.plot(k, [e["sigma"] for e in eps], color=ACCENT, marker="^", ms=4,
            label=r"mean GP $\sigma$")
    ax.set_xticks(k)
    ax.legend(fontsize=8)
    _tidy(ax, "episode", None, "Less uncertainty, less detour")
    fig.tight_layout()
    return _save(fig, "fig11_gp_margin")


# ============================================================================
# FIG 12 -- CBF filtering inside a policy-search loop (safe exploration)
# ============================================================================

def fig12():
    """A deliberately small safe-RL experiment.

    Cross-entropy-method policy search on a 2-D navigation task, run twice:
    once with raw policy actions, once with every action passed through a
    closed-form CBF projection during *training*.  We count safety violations
    accumulated while learning -- the quantity that makes or breaks learning on
    real hardware, and the motivation for RL-CBF, RCBF-SAC and CBF-RL.
    """
    rng = np.random.default_rng(4)
    obst = [(np.array([-0.6, 0.35]), 0.75), (np.array([1.3, -0.55]), 0.65)]
    goal, x0, u_max, dt, H = np.array([3.0, 1.2]), np.array([-3.0, -1.0]), 1.0, 0.06, 160
    gamma_cbf = 2.0

    def features(p):
        e = goal - p
        ne = np.linalg.norm(e) + 1e-9
        f = [e[0] / ne, e[1] / ne]
        for c, r in obst:
            d = p - c
            nd = np.linalg.norm(d) + 1e-9
            f += [d[0] / nd / (1 + nd), d[1] / nd / (1 + nd)]
        return np.array(f + [1.0])

    def cbf_project(p, u):
        """Closed-form projection onto the most-binding half-space."""
        hs = [(np.linalg.norm(p - c) - r, (p - c) / (np.linalg.norm(p - c) + 1e-9))
              for c, r in obst]
        h, g = min(hs, key=lambda z: z[0])
        v = g @ u + gamma_cbf * h
        if v < 0:
            u = u + g * (-v)
        n = np.linalg.norm(u)
        return u * (u_max / n) if n > u_max else u

    def rollout(W, use_filter):
        p, R, viol = x0.copy(), 0.0, 0
        for _ in range(H):
            u = u_max * np.tanh(W @ features(p))
            if use_filter:
                u = cbf_project(p, u)
            p = p + dt * u
            R -= 0.03 * np.linalg.norm(goal - p)
            if any(np.linalg.norm(p - c) < r for c, r in obst):
                viol += 1
                R -= 2.0
            if np.linalg.norm(goal - p) < 0.2:
                R += 25.0
                break
        R -= 1.5 * np.linalg.norm(goal - p)
        return R, viol, p

    def cem(use_filter, iters=25, pop=40, elite=8, seed=1):
        r = np.random.default_rng(seed)
        nf = len(features(x0))
        mu, sd = np.zeros((2, nf)), 0.8 * np.ones((2, nf))
        curve, viols, best = [], [], (-1e9, None)
        for it in range(iters):
            Ws = mu[None] + sd[None] * r.standard_normal((pop, 2, nf))
            res = [rollout(W, use_filter) for W in Ws]
            Rs = np.array([z[0] for z in res])
            viols.append(sum(z[1] for z in res))
            idx = np.argsort(-Rs)[:elite]
            mu, sd = Ws[idx].mean(0), Ws[idx].std(0) + 0.02
            if Rs[idx[0]] > best[0]:
                best = (Rs[idx[0]], Ws[idx[0]].copy())
            curve.append(Rs[idx].mean())
        return np.array(curve), np.array(viols), best[1]

    c_raw, v_raw, W_raw = cem(False)
    c_flt, v_flt, W_flt = cem(True)

    def traj(W, use_filter):
        p, xs = x0.copy(), [x0.copy()]
        for _ in range(H):
            u = u_max * np.tanh(W @ features(p))
            if use_filter:
                u = cbf_project(p, u)
            p = p + dt * u
            xs.append(p.copy())
            if np.linalg.norm(goal - p) < 0.2:
                break
        return np.array(xs)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    ax = axes[0]
    ax.plot(np.cumsum(v_raw), color=UNSAFE, label="unfiltered policy search")
    ax.plot(np.cumsum(v_flt), color=FILTERED, label="CBF-filtered exploration")
    ax.legend(fontsize=8.4)
    _tidy(ax, "CEM iteration", "cumulative unsafe steps",
          "Violations accumulated while learning")

    ax = axes[1]
    ax.plot(c_raw, color=UNSAFE)
    ax.plot(c_flt, color=FILTERED)
    _tidy(ax, "CEM iteration", "elite mean return", "Task performance")

    ax = axes[2]
    for c, r in obst:
        ax.add_patch(Circle(c, r, facecolor=UNSAFE, alpha=0.15,
                            edgecolor=UNSAFE, lw=1.4))
    ax.plot(*traj(W_raw, False).T, color=UNSAFE, label="learned, unfiltered")
    ax.plot(*traj(W_flt, True).T, color=FILTERED, label="learned + filter")
    ax.plot(*goal, "*", color=INK, ms=11)
    ax.set_aspect("equal"); ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.2, 2.2)
    ax.legend(fontsize=8.0, loc="lower right")
    _tidy(ax, r"$p_1$", r"$p_2$", "Best policy found")
    fig.tight_layout()
    return _save(fig, "fig12_safe_rl")


# ============================================================================
# Registry + CLI
# ============================================================================

def _registry():
    return {k: v for k, v in sorted(globals().items())
            if k.startswith("fig") and callable(v)}


if __name__ == "__main__":
    reg = _registry()
    names = sys.argv[1:] or list(reg)
    for n in names:
        if n not in reg:
            print("unknown figure:", n, "-- available:", ", ".join(reg))
            continue
        print("==", n)
        reg[n]()
