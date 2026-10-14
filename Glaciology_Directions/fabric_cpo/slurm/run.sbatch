#!/bin/bash
#SBATCH --job-name=fabric
#SBATCH --account=envi.p_forclima
#SBATCH --partition=smp
#SBATCH --qos=30min
#SBATCH --time=00:29:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/fabric_cpo/slurm/logs/%x_%j.out
#SBATCH --error=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/fabric_cpo/slurm/logs/%x_%j.out
#
# usage (on albedo):
#   sbatch slurm/run.sbatch test1_ranks.jl cases="5,5,0.2"
#   sbatch --qos=12h --time=02:00:00 slurm/run.sbatch stepper.jl ...
set -e
REPO=/albedo/home/taange001/CFD_with_MPS
cd $REPO/Glaciology_Directions/fabric_cpo
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GKSwstype=100
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)  args: $@"
SCRIPT=$1; shift
/usr/bin/time -f "elapsed %e s, maxrss %M kB" julia +1.12.7 --project=$REPO $SCRIPT "$@"
