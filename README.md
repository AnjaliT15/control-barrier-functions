# Control barrier functions — code for the blog post

Simulation code for the post
[*Control barrier functions: theory, variants, and learning-based extensions*](https://anjalit15.github.io/blog/).
Every figure in that post is produced by this code; nothing is hand-drawn.

## Files

| File | Contents |
|---|---|
| `cbf_core.py` | The toolbox: CBF-QP safety filter (with optional slack), a conic filter for measurement-robust / actuation-uncertain conditions, HOCBF constraint rows for relative degree 2, extended class-K functions, barrier functions, log-sum-exp aggregation, and a zero-order-hold simulator. numpy + scipy only. |
| `cbf_examples.py` | The twelve experiments, one function per figure. Run the whole set or any subset. |
| `learning_cbf_reference.py` | Reference implementations for the learning sections: a neural barrier with the three standard losses, a counterexample-guided (CEGIS) training loop, verification notes, a differentiable CBF layer, and a Gym wrapper for filtering actions during RL training. Requires PyTorch; not needed for the figures. |

## Running it

```bash
pip install -r requirements.txt

python cbf_examples.py              # regenerate every figure into ./figs
python cbf_examples.py fig05 fig09  # or just the ones you want
```

Each `figNN` function is self-contained and commented so it can be read next to
the corresponding section of the post and edited freely.

## Which figure comes from which function

| Figure | Function | Demonstrates |
|---|---|---|
| 1 | `fig01` | Forward invariance; the decay envelope of the CBF inequality |
| 2 | `fig03` | The CBF-QP as a projection in input space |
| 3 | `fig02` | Class-K choice and conservatism |
| 4 | `fig04` | Adaptive cruise control with actuator limits |
| 5 | `fig05` | Relative degree 2: naive CBF vs HOCBF |
| 6 | `fig06` | Robust and ISSf conditions under drift |
| 7 | `fig08` | Deadlock at a stable undesired equilibrium |
| 8 | `fig09` | Zero-order-hold erosion and the sampled-data margin |
| 9 | `fig07` | Decentralised pairwise CBFs and the neighbour graph |
| 10 | `fig10` | A barrier learned from labelled samples (kernel SVM) |
| 11 | `fig11` | GP-learned dynamics and a shrinking robust margin |
| 12 | `fig12` | CBF-filtered policy search: violations accumulated during learning |

## Notes

The QP is solved with SLSQP so that the only dependencies are numpy and scipy.
For anything real-time, swap `solve_qp` for a dedicated QP solver (`qpOASES`,
`OSQP`, `proxQP`); the interface is small enough to make that a drop-in change.

Related implementations worth starting from rather than rewriting:
[`MIT-REALM/gcbfplus`](https://github.com/MIT-REALM/gcbfplus) for graph CBFs,
`FOSSIL` for formal synthesis of barrier certificates, and the
alpha-beta-CROWN toolchain for verifying learned barriers.
