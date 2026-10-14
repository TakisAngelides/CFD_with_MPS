# AMEn linear solver for A x = b in MPS/MPO form (Dolgov & Savostyanov, SISC 36 (2014) A2248):
# single-site ALS updates of x (dense LU of the chi*d*chi local system, 8x cheaper than the
# two-site DMRG solve of ITensorMPS.linsolve at the same bond dimension) plus rank adaptivity by
# enriching each updated core with the projected residual. The residual is carried as a second,
# low-rank MPS z (rank `kick`) updated by its own ALS projection, so the enrichment directions
# point at what the current x cannot yet represent instead of at random.
# Environments (all built with the dagger on the primed side):
#   XAX: x'^+ A x    Xb: x'^+ b    ZAX: z'^+ A x    Zb: z'^+ b
using ITensors, ITensorMPS, LinearAlgebra, Random


struct AmenEnv
    XAX::Vector{ITensor}; Xb::Vector{ITensor}; ZAX::Vector{ITensor}; Zb::Vector{ITensor}
end
AmenEnv(n) = AmenEnv([ITensor(1.0) for _ in 1:n+2], [ITensor(1.0) for _ in 1:n+2], [ITensor(1.0) for _ in 1:n+2], [ITensor(1.0) for _ in 1:n+2])

# environment of sites 1..k (left, stored at k+1) or k..n (right, stored at k+1); slot 1 / n+2 are trivial
function _env_step!(E::AmenEnv, A, b, x, z, k, slot_from, slot_to)
    xp = dag(prime(x[k])); zp = dag(prime(z[k]))
    E.XAX[slot_to] = E.XAX[slot_from] * x[k] * A[k] * xp
    E.Xb[slot_to] = E.Xb[slot_from] * b[k] * dag(prime(x[k], "Link"))
    E.ZAX[slot_to] = E.ZAX[slot_from] * x[k] * A[k] * zp
    E.Zb[slot_to] = E.Zb[slot_from] * b[k] * dag(prime(z[k], "Link"))
end

# local dense solve of (L A_k R) x_k = (L b_k R); GMRES above maxdirect
function _local_solve(L, Ak, R, f, xk; maxdirect = 8000)
    if dim(xk) <= maxdirect
        C = combiner(inds(xk)...); c = combinedind(C)
        M = matrix(((L * Ak) * R) * C * prime(C), prime(c), c)
        rhs = vector(noprime(f) * C)
        F = lu(M; check = false)
        xv = issuccess(F) ? F \ rhs : qr(M, ColumnNorm()) \ rhs
        return itensor(xv, c) * dag(C)
    end
    op(v) = noprime(((L * v) * Ak) * R)
    xs, _ = linsolve(op, noprime(f), xk; krylovdim = 40, maxiter = 20, tol = 1e-10)
    return xs
end

# A::MPO, b::MPS, x0::MPS on the same sites. Returns x and (relative residual, sweeps).
function amen_solve(A::MPO, b::MPS, x0::MPS; nsweeps = 10, tol = 1e-8, cutoff = 1e-10, maxdim = 64, kick = 4,
                    maxdirect = 8000, verbose = false)
    n = length(A)
    x = sim(linkinds, copy(x0)); orthogonalize!(x, 1)
    z = random_mps(siteinds(x); linkdims = kick); orthogonalize!(z, 1)
    E = AmenEnv(n)
    for k in n:-1:2
        _env_step!(E, A, b, x, z, k, k + 2, k + 1)
    end
    nb = norm(b); res = Inf; sw = 0
    for s in 1:nsweeps
        sw = s
        dir = isodd(s) ? 1 : -1
        for k in (dir == 1 ? (1:n) : (n:-1:1))
            # slots: left env of site k is E[k], right env is E[k+2]
            Lx, Rx = E.XAX[k], E.XAX[k+2]
            f = E.Xb[k] * b[k] * E.Xb[k+2]
            xk = _local_solve(Lx, A[k], Rx, f, x[k]; maxdirect)
            x[k] = xk
            j = k + dir
            (j < 1 || j > n) && continue
            bond = commonind(x[k], x[j])
            uinds = uniqueinds(x[k], x[j])
            # truncate the new core
            U, S, V = svd(xk, uinds; cutoff, maxdim, use_relative_cutoff = true)
            xt = U * S * V
            # projected residuals: z-core (z interfaces both sides) and the x-enrichment (x on the inner side)
            if dir == 1
                yk = noprime(E.Xb[k] * b[k] * E.Zb[k+2]) - noprime(((E.XAX[k] * A[k]) * xt) * E.ZAX[k+2])
            else
                yk = noprime(E.Zb[k] * b[k] * E.Xb[k+2]) - noprime(((E.ZAX[k] * A[k]) * xt) * E.XAX[k+2])
            end
            zk = noprime(E.Zb[k] * b[k] * E.Zb[k+2]) - noprime(((E.ZAX[k] * A[k]) * xt) * E.ZAX[k+2])
            # x <- [U, y] orthonormalised; the next core absorbs the projection of the truncated x_k
            ui = commonind(U, S)
            zb = commonind(z[k], z[j])
            yk = replaceind(yk, zb, sim(zb))
            yb = uniqueind(yk, U)
            Cc, cind = directsum(U => ui, yk => yb)
            Q, _ = qr(Cc, uinds; tags = tags(bond))
            x[k] = Q
            x[j] = (dag(Q) * xt) * x[j]
            # z <- leading kick directions of its projected residual
            zuinds = uniqueinds(z[k], z[j])
            Uz, Sz, Vz = svd(zk, zuinds; maxdim = kick)
            Qz, _ = qr(Uz, zuinds; tags = tags(zb))
            z[j] = (dag(Qz) * zk) * z[j]
            z[k] = Qz
            dir == 1 ? _env_step!(E, A, b, x, z, k, k, k + 1) : _env_step!(E, A, b, x, z, k, k + 2, k + 1)
        end
        res = residual_norm(A, x, b)
        verbose && println("    amen sweep $s: res $(round(res, sigdigits = 3)) chi $(maxlinkdim(x))")
        res < tol && break
    end
    return x, (res, sw)
end
