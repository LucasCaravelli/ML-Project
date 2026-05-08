#!/bin/bash
#SBATCH --account=3404007
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --job-name=fusion
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

# Evaluates RRF fusion on the test split. Uses cluster.yaml (vLLM + Qwen2.5-7B)
# to stay apples-to-apples with the baselines and oracle in oracle_gap.json,
# which were computed under the same backend.
python -m experiments.run_fusion --config configs/cluster.yaml
