#!/bin/bash
#SBATCH --job-name=dense_dt
#SBATCH --account=envi.p_forclima
#SBATCH --partition=smp
#SBATCH --qos=30min
#SBATCH --time=00:29:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/albedo/home/taange001/CFD_with_MPS/Glaciology_SIA/slurm/logs/%x_%j.out
# dt / Picard-iteration convergence of the dense 1D reference scheme
cd /albedo/home/taange001/CFD_with_MPS/Glaciology_SIA
for dt in 1 10 100; do for it in 3 30; do
  echo "### dt=$dt iters=$it"
  julia +1.12.7 --project=.. dense_sia_1d.jl nbits=12 scheme=be dt=$dt t_run=3000 max_iters=$it tol=1e-10 outdir=/tmp/dense_$SLURM_JOB_ID | grep -E "^step" | tail -1
done; done
