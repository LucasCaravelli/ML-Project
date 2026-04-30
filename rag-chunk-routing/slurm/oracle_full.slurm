#!/bin/bash
#SBATCH --account=3404007
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --job-name=oracle_full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$HOME/projects/ML-Project/rag-chunk-routing"
module load miniconda3
source activate ragcr
export HF_HOME="$HOME/.cache/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nvidia-smi

# Adds train + val rows to the existing eval grid (test rows from oracle_test.slurm
# are preserved by append-merge in compute_oracle.py).
python -m experiments.compute_oracle --config configs/cluster.yaml --splits train,val
python -m experiments.run_baselines --config configs/cluster.yaml
