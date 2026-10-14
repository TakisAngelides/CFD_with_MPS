# Instantiate + precompile the project environment on the cluster (run once / after Manifest changes)
using Pkg
Pkg.instantiate()
Pkg.precompile()
