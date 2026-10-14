#!/bin/bash
#SBATCH --job-name=qttc
#SBATCH --account=envi.p_forclima
#SBATCH --partition=smp
#SBATCH --qos=30min
#SBATCH --time=00:29:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/data_compression/logs/%x_%j.out
#SBATCH --error=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/data_compression/logs/%x_%j.out
#
# usage: sbatch [--qos=12h --time=..] [--job-name=qttc_x] slurm/job.sbatch script.jl key=value ...
set -e
DIR=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/data_compression
cd $DIR
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)  args: $@"
SCRIPT=$1; shift
/usr/bin/time -f "elapsed %e s, maxrss %M kB" julia +1.12.7 --project=/albedo/home/taange001/CFD_with_MPS $SCRIPT "$@"
