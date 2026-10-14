# Glaciology_SIA — Shallow Ice Approximation with quantics MPS

Isothermal SIA ice flow (mass balance + SIA velocity) solved directly in quantics tensor-train / MPS
form with ITensors, in 1D (flowline) or 2D (map plane), with implicit time stepping that has no CFL
restriction. Built on the same machinery as `Burgers_Equation/` (OpSum shift strings, diagonal-MPO
Hadamard products, TCI from tensor4all), generalised to d dimensions.

## Equations

    dH/dt = a - div(H u)                                 mass balance
    u     = -Gamma H^(n+1) |grad s|^(n-1) grad s         SIA velocity,  s = H + b
    Gamma = 2 A (rho g)^n / (n + 2)

Velocity and mass balance are coupled through the diffusivity D = H k, k = Gamma H^(n+1)|grad s|^(n-1),
so div(H u) = -div(D grad s). Glen's n must be odd so that every nonlinearity is a polynomial,
i.e. an exact product of MPS (n = 3 by default).

## Discretisation

* **Grid / encoding** (`quantics.jl`): `QuanticsGrid(nbits, ordering, (x0,x1), (y0,y1), ...)`,
  any number of dimensions, `2^nbits` cell-centred points per dimension, `d*nbits` MPS sites.
  `ordering = :interleaved` (x1 y1 x2 y2 ...) or `:sequential` (x1 ... xn y1 ... yn).
  Every operator is built from `grid.pos[dim]`, so dimension and ordering are pure options.
* **Operators**: shifts along any dimension (`shift_mpo`, open or periodic, bond dim 3), forward /
  backward differences, face averages, Hadamard products/powers, TCI loading of functions
  (`function_to_mps`), exact loading of arrays, fast point evaluation (`MPSEvaluator`), elementwise
  nonlinear maps by TCI re-compression (`map_mps`, used for H <- max(H, 0)).
* **Space** (`sia.jl`): conservative staggered Mahaffy (type II) stencil, D on faces.
* **Time**: backward Euler or BDF2 (`scheme`, variable-step BDF2), both A-stable, with adaptive
  dt: a step whose nonlinear solve fails is retried with dt/2, easy steps grow dt by 1.5x up to the
  requested `dt`. Nonlinear solve by
  * Newton (`nonlinear=newton`, default): full SIA Jacobian assembled as one MPO (verified against
    finite differences), backtracking line search, convergence judged on the full Newton correction
    (`tol`), and accepted at the residual noise floor (`noise_tol`) when no descent is possible;
  * Picard (`nonlinear=picard`): (I - dt L[D(H^m)]) H^(m+1) = rhs, L[D] = sum_a Db_a diag(D_a) Df_a,
    SPD for D >= 0. Only adequate at coarse resolution, see findings.
  Linear systems: `linear_solver=linsolve` (default, DMRG-like two-site MPS solve with the local
  problems solved exactly by dense LU, `local_solver=direct`, or by KrylovKit, `local_solver=krylov`;
  residual-checked) or `linear_solver=inverse` (warm-started Newton-Schulz MPO inverse).
* **Ice-free cells with a mass balance (EISMINT)**: H >= 0 is imposed as a complementarity problem
  H >= 0, R(H) >= 0, H R(H) = 0, solved by semismooth Newton on Phi = min(H, R) (`complementarity`,
  default for EISMINT): rows in the active set (H = 0 enforced) become identity rows,
  J = I - dt diag(mask) dF/dH. min(H, R) and the 0/1 mask are TCI maps of two MPS (a 0/1 mask
  has QTT rank ~2 per switching point in 1D). Without it, ice-free ablation zones get H = dt a < 0
  (negative D ~ H^5, anti-diffusion). The simpler `limit_ablation` (rhs = max(H^n + dt a, 0)) is
  dt-dependent: with dt ~ 600 a it suppresses ablation wherever H < |a| dt ~ 1 km and moved the
  steady 1D margin from 655 to 668 km.

## Benchmarks (`benchmarks.jl`)

* **Halfar / Bueler et al. (2005) Test B** — exact similarity solution, H0 = 3600 m, R0 = 750 km,
  A = 1e-16 Pa^-3 a^-1, n = 3, flat bed, a = 0. Generalised to d = 1 and 2. Exact errors, volume.
* **EISMINT I moving margin** (Huybrechts et al. 1996) — flat bed, 1500 km box,
  a = min(0.5, 1e-5 (450 km - r)) m/a, integrate to steady state. Steady margin from
  mass-balance closure: 579.8 km.

## Running (on albedo, via Slurm)

    cd ~/CFD_with_MPS/Glaciology_SIA
    sbatch --qos=30min --time=00:29:00 slurm/run.sbatch setup.jl     # once: instantiate + precompile
    sbatch --qos=30min --time=00:29:00 slurm/run.sbatch tests.jl     # checks against dense arrays
    sbatch slurm/run.sbatch run_sia.jl experiment=halfar d=2 nbits=8 dt=100 t_run=20000
    sbatch slurm/run.sbatch run_sia.jl experiment=eismint d=2 nbits=8 dt=200 t_run=50000
    sbatch --qos=30min --time=00:29:00 slurm/run.sbatch plot_sia.jl output/<file>.h5

All options of `run_sia.jl` are `key=value` arguments (see `cfg` at the top of the script).
Output: `output/*.h5` (time series of errors, bond dimensions, wall times, sampled snapshots,
final speed) and PNGs from `plot_sia.jl`.

## Findings so far

**Precision at high resolution (the main lesson).** The SIA operator at fine resolution is a
stiff, degenerate operator (cond(I - dt J) ~ 1e5 at 2^14, 2e6 at 2^16, 2e8 at 2^18 points for
dt = 2.5 a, smallest singular vector in the margin cell). Four default behaviours of the MPS
toolchain each silently destroyed the margin and had to be replaced (`bench_precision.jl`,
`bench_jacobian.jl`, `bench_cond16.jl`):
1. MPO x MPS products: ITensors' default `apply` is the density-matrix algorithm, precision floor
   ~ sqrt(eps) ~ 1e-8 relative -> exact contraction + SVD (`mps_apply`, `PRODUCT_ALG = "naive"`).
2. MPS/MPO sums: ITensors' default `+` is also density-matrix based with a fixed cutoff 1e-15 ->
   exact direct sum + SVD (`exact_sum`, used by `lincomb`, `mpo_lincomb`, all operator assembly).
   Passing a tighter cutoff to the default algorithms helps but plateaus (`bench_sum.jl`, error vs
   the sparse value of the same step):

   | points | flux, apply cutoff 1e-15 / 1e-30 / exact | residual, add default / cutoff 1e-30 / direct sum |
   |---|---|---|
   | 2^12 | 1.2e-3 / 3.2e-4 / 1.0e-7 | 2.7e-4 / 5.7e-5 / 2.8e-7 |
   | 2^14 | 1.5e-2 / 7.8e-4 / 4.5e-7 | 2.0e-4 / 2.3e-5 / 1.1e-6 |
   | 2^16 | 1.0e-1 / 4.5e-3 / 5.3e-6 | 9.1e-5 / 1.7e-5 / - |
3. Truncation inside the nonlinear products is relative to the norm of each intermediate. H^5 is
   dominated by the 3.6 km dome, |grad s|^2 by the margin, so forming H^5 first leaves its values
   in the thin margin cells (H^5 there ~ 1e-10 of the dome value at 2^16) below the truncation
   level. Products are therefore formed gradients first (`mul_pow`: |grad s|^2 H H H ...), so every
   intermediate is balanced, with a tight product cutoff (`prod_cutoff = 1e-24`).
4. The residual ||b - A x|| expanded as <b,b> - 2<b|A|x> + <Ax|Ax> cancels catastrophically once
   sigma_max(A) ~ 1e6 -> exact b - A x, orthogonalized before taking the norm (`residual_norm`).
5. The state itself: the residual of a step evaluated on a truncated H has a noise floor
   ~ H (D_max/h^2) dt sqrt(cutoff). Tendencies below |dH/dt| ~ H (D_max/h^2) sqrt(cutoff) are
   invisible (5 cm/a for cutoff 1e-13 at 2^10 EISMINT): the approach to steady state froze (dome
   stuck 7 m low, margin 6 km short) and steps were rejected down to dt ~ 10 a. With cutoff 1e-20
   (chi(H) 10 -> 14) dt grows to 500-700 a near steady state.
Plus a tight operator cutoff (`op_cutoff = 1e-22`). Effect at 2^16 points: margin flux error 100 %
-> 3e-6, Newton update vs sparse direct solve 280 % -> 1.5e-5 (2^18: 1e-4).

**Linear solver: exact local solves.** The DMRG-like `linsolve` solves a small local problem per
bond; with the default local GMRES these inherit the global conditioning and stall or break down
(`SingularException`). Assembling the projected two-site operator as a dense matrix and solving it
by LU (`direct_updater`, local size 4 chi^2 ~ 400-4000 in 1D) gives the same answer 25-250x faster
(2^12 Newton system: 0.7 s instead of 18 s) and works where GMRES failed. The Newton update only
needs ~ tol relative accuracy, so it is truncated more loosely (`lin_cutoff`, `lin_maxdim`).

**Newton solvability vs. margin motion.** Stability is unconditional, but Newton advances a
degenerate front only about one cell per iteration (D = 0 ahead of the margin), so the step must
not move the margin many cells. The adaptive dt finds this limit automatically; for Halfar at
t0 (margin speed ~ 97 m/a) it settles at ~2-3 cells per step. Large steps remain possible where
the margin is slow (later Halfar times, approach to EISMINT steady state).

**Bond dimensions** (`bench_ranks.jl`, Halfar at t0, dt = 100 a, operator A = I - dt L[D] exact up to 1e-15):

| d | points | ordering | chi(H) | chi(D_x) | chi(A) |
|---|---|---|---|---|---|
| 1 | 2^10 ... 2^20 | — | 10-11 | 14-15 | 17-18 (flat in resolution) |
| 2 | 64^2 | interleaved / sequential | 32 / 32 | 32 / 32 | 81 / 65 |
| 2 | 128^2 | interleaved / sequential | 64 / 58 | 63 / 57 | 128 / 118 |
| 2 | 256^2 | interleaved / sequential | 104 / 92 | 113 / 99 | 230 / 201 |
| 2 | 512^2 | interleaved / sequential | 134 / 162 | 177 / 173 | 377 / 348 |

* 1D (flowline): ranks are independent of resolution up to 2^20 points, i.e. the cost of a step
  grows only ~linearly in the number of bits — the exponential MPS advantage.
* 2D (map plane): ranks grow ~ 2^(n/2) because the ice margin is a curved front with a slope
  singularity (H ~ (R-r)^(3/7)), the generic QTT behaviour for features along a curve. Loosening the
  TCI tolerance from 1e-8 to 1e-4 lowers chi(H) only modestly (104 -> 71 at 256^2), chi(D), chi(A)
  hardly change. Sequential ordering gives ~10-15 % smaller operators up to 256^2 for this radial case.
  In the 64^2 Halfar runs (BDF2, dt = 100 a, Newton) sequential ordering gives chi(A) = 96 vs 133
  and ~17 s vs ~57-80 s per step, at the same accuracy (dome error ~3.5 m, volume error 1e-3).
  Possible remedies (not implemented): coordinates aligned with the margin (polar / mapped grids),
  smoothing the margin, or a level-set treatment of the front.

**Nonlinear solver: Picard fails at fine resolution, Newton works** (`dense_sia_1d.jl`, same
discrete scheme with sparse matrices). At the Halfar margin the slope is infinite, so the face
diffusivity grows with resolution (~1e10 m^2/a at 586 m spacing, explicit limit ~1e-5 a). A frozen-D
(Picard) solve flattens the margin zone, D ~ |grad s|^2 then vanishes there and Picard oscillates
without converging: margin and dome stall (1D, 2^12 points, dt = 10 a, after 3 ka: dome 3600 m vs
exact 3091 m, margin 752 km vs 873 km) — independent of dt and of MPS. Damped Newton converges
quadratically (4-9 iterations to residual 1e-13) and reproduces the exact solution: dome 3092.1 m,
margin 873.6 km, max error ~20 m (all at the margin), at dt = 10 a ~ 1e6 x the explicit limit.
At dt = 100 a the margin would have to jump ~14 cells per step and Newton does not converge
(a solvability, not a stability limit) -> automatic step halving. The 64^2 2D runs were fine with
Picard only because the coarse grid keeps the margin diffusivity moderate.

**Linear solver** (`bench_linsolve.jl`, `bench_inverse.jl`): cond(I - dt L) grows 4x per bit
(3.6e3 at 2^10, 9.3e5 at 2^14 for dt = 100 a).
* `linsolve` (DMRG-like, GMRES locally) stays usable: ~10-20 s per solve in 1D up to 2^14, relative
  error vs a sparse direct solve 2e-6 (2^10) ... 6e-4 (2^14, ~2 m on a 3.6 km dome, far below the SIA
  discretisation error at the margin). Local CG is faster only at low resolution.
* Occasional local-solve failures (residual O(0.1-10) instead of ~3e-3) used to destroy runs
  (negative H -> negative D -> loss of definiteness); the solve is now checked with an exact
  residual (no truncation noise) and re-swept if needed.
* `linear_solver=inverse` (Newton-Schulz tracked MPO inverse, the MPO-inverse idea of the Burgers
  code without dense local SVDs): converges in ~12 iterations at 2^8 with chi(X) ~ 50, but the inverse
  of this degenerate variable-coefficient operator is not low rank (chi(X) > 256 at 2^10), so it is
  an option, not the default.

## Files

| file | content |
|---|---|
| `quantics.jl` | d-dimensional quantics grid, orderings, shift MPOs, Hadamard algebra, TCI/array I/O |
| `sia.jl` | SIA velocity, face diffusivities, operator/Jacobian MPOs, implicit BE/BDF2 step |
| `benchmarks.jl` | Halfar/Bueler B exact solution, EISMINT I moving-margin setup |
| `run_sia.jl` | driver, writes HDF5 |
| `plot_sia.jl` | headless plots |
| `tests.jl` | consistency checks vs dense arrays |
| `bench_linsolve.jl` | linear-solver settings vs sparse direct solve |
| `bench_inverse.jl` | Newton-Schulz MPO inverse vs linsolve vs sparse direct |
| `bench_ranks.jl` | bond dimensions vs resolution / ordering / cutoff |
| `slurm/run.sbatch` | albedo job wrapper |
