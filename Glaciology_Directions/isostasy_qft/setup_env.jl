# Run once inside a Slurm job: makes ./env a copy of the CFD_with_MPS environment (same versions,
# so the precompiled caches are reused) plus SpecialFunctions and QuadGK (already in that manifest).
using Pkg
dir = @__DIR__
env = joinpath(dir, "env")
mkpath(env)
for f in ("Project.toml", "Manifest.toml", "LocalPreferences.toml")
    cp(joinpath(dir, "..", "..", f), joinpath(env, f); force = true)
end
Pkg.activate(env)
Pkg.offline(true)
Pkg.add(["SpecialFunctions", "QuadGK"]; preserve = Pkg.PRESERVE_ALL)
Pkg.instantiate()
Pkg.precompile()
Pkg.status()
using ITensors, ITensorMPS, TensorCrossInterpolation, TCIITensorConversion, FFTW, HDF5, SpecialFunctions, QuadGK
println("env ok: besselj1(1.0) = ", besselj1(1.0))
