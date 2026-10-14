#!/bin/bash
#SBATCH --job-name=mps_sia
#SBATCH --account=envi.p_forclima
#SBATCH --partition=smp
#SBATCH --qos=12h
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/albedo/home/taange001/CFD_with_MPS/Glaciology_SIA/slurm/logs/%x_%j.out
#SBATCH --error=/albedo/home/taange001/CFD_with_MPS/Glaciology_SIA/slurm/logs/%x_%j.out
#
# usage (from anywhere on albedo):
#   sbatch slurm/run.sbatch run_sia.jl experiment=halfar d=2 nbits=9 dt=100
#   sbatch slurm/run.sbatch tests.jl
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch run_sia.jl d=1 t_run=1000
# the first argument is the script in Glaciology_SIA, the rest are passed on as key=value options
set -e
REPO=/albedo/home/taange001/CFD_with_MPS
cd $REPO/Glaciology_SIA
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)  args: $@"
echo "repo: $(cd $REPO && git log --oneline -1) (+ uncommitted: $(cd $REPO && git status --short | wc -l) files)"
SCRIPT=$1; shift
/usr/bin/time -f "elapsed %e s, maxrss %M kB" julia +1.12.7 --project=$REPO $SCRIPT "$@"
