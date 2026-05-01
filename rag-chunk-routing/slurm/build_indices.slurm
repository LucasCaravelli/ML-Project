#!/bin/bash
#SBATCH --account=3404007
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --job-name=build_indices
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$HOME/projects/ML-Project/rag-chunk-routing"
module load miniconda3
source activate ragcr
export HF_HOME="$HOME/.cache/huggingface"

# Use base.yaml (CPU embeddings) since this job doesn't request a GPU.
python -m experiments.build_indices --config configs/base.yaml
