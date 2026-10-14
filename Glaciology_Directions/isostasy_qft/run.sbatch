#!/bin/bash
#SBATCH --job-name=iso_qft
#SBATCH --account=envi.p_forclima
#SBATCH --partition=smp
#SBATCH --qos=30min
#SBATCH --time=00:29:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/isostasy_qft/logs/%x_%j.out
#SBATCH --error=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/isostasy_qft/logs/%x_%j.out
#
# usage (on albedo, from anywhere):
#   sbatch run.sbatch t1_qft.jl [key=value ...]
#   PROJ=/albedo/home/taange001/CFD_with_MPS sbatch run.sbatch script.jl   (use the main project)
#   sbatch --qos=12h --time=04:00:00 --mem=128G run.sbatch script.jl
# Default project: ./env (copy of the CFD_with_MPS manifest + SpecialFunctions, QuadGK)
set -e
DIR=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/isostasy_qft
cd $DIR
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_PKG_OFFLINE=true
PROJ=${PROJ:-$DIR/env}
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)  project $PROJ  args: $@"
SCRIPT=$1; shift
/usr/bin/time -f "elapsed %e s, maxrss %M kB" julia +1.12.7 --project=$PROJ $SCRIPT "$@"
