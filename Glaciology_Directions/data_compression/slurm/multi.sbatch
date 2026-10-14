#!/bin/bash
#SBATCH --job-name=qttc_multi
#SBATCH --account=envi.p_forclima
#SBATCH --partition=smp
#SBATCH --qos=12h
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/data_compression/logs/%x_%j.out
#SBATCH --error=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/data_compression/logs/%x_%j.out
#
# usage: sbatch --job-name=qttc_A slurm/multi.sbatch "field=grl_bed task=rd n=13" "field=grl_thk task=rd n=13" ...
# runs run_compress.jl once per quoted argument, sequentially (a failure does not stop the rest)
DIR=/albedo/home/taange001/CFD_with_MPS/Glaciology_Directions/data_compression
cd $DIR
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)"
for item in "$@"; do
  echo "######## $item   $(date)"
  /usr/bin/time -f "elapsed %e s, maxrss %M kB" julia +1.12.7 --project=/albedo/home/taange001/CFD_with_MPS run_compress.jl $item || echo "######## FAILED: $item"
done
echo "all items finished $(date)"
