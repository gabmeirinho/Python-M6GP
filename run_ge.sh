#!/bin/bash
#SBATCH --job-name=m6gp_classification
#SBATCH --output=output/output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

uv run python3 -u main.py