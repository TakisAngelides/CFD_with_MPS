#!/bin/bash
#SBATCH --job-name=hydro_shakti
#SBATCH --account=envi.p_forclima
#SBATCH --partition=smp
#SBATCH --qos=30min
#SBATCH --time=00:29:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/hydrology_shakti/logs/%x_%j.out
#SBATCH --error=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/hydrology_shakti/logs/%x_%j.out
# Shakti runs with the user's Shakti.jl environment, read-only (julia release channel = the
# version its Manifest was resolved with). Usage: sbatch [--qos/--time overrides] shakti.sbatch script.jl args...
set -e
DIR=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/hydrology_shakti
cd $DIR
export OPENBLAS_NUM_THREADS=${BLAS_THREADS:-1}   # 1: CHOLMOD on 2D grids gains nothing from threaded BLAS, and spinning BLAS threads slowed Shakti 4-7x
JT=${JULIA_THREADS:-$SLURM_CPUS_PER_TASK}
echo "julia threads $JT, openblas threads $OPENBLAS_NUM_THREADS"
echo "host $(hostname) job $SLURM_JOB_ID $(date) args: $@"
SCRIPT=$1; shift
/usr/bin/time -f "elapsed %e s, maxrss %M kB" julia +release -t $JT --project=/albedo/home/taange001/Shakti.jl $SCRIPT "$@"
