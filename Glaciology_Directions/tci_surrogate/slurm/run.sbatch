#!/bin/bash
#SBATCH --job-name=tci_sur
#SBATCH --account=envi.p_forclima
#SBATCH --partition=smp
#SBATCH --qos=30min
#SBATCH --time=00:29:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/tci_surrogate/slurm/logs/%x_%j.out
#SBATCH --error=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/tci_surrogate/slurm/logs/%x_%j.out
#
# usage: sbatch [--qos=12h --time=HH:MM:SS] [-J name] slurm/run.sbatch script.jl key=value ...
# Julia threads parallelise model runs; each run is single-threaded (BLAS threads = 1).
set -e
REPO=/albedo/home/taange001/CFD_with_MPS
cd $REPO/Glaciology_Directions/tci_surrogate
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export GKSwstype=100
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)  threads $SLURM_CPUS_PER_TASK  args: $@"
SCRIPT=$1; shift
/usr/bin/time -f "elapsed %e s, maxrss %M kB" julia +1.12.7 --project=$REPO $SCRIPT "$@"
