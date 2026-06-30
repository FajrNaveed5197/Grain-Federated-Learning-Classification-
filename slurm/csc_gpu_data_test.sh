#!/bin/bash
#SBATCH --job-name=grain_gpu_check
#SBATCH --account=project_2019649
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2019649/grain_research/logs/grain_gpu_check_%j.out
#SBATCH --error=/scratch/project_2019649/grain_research/logs/grain_gpu_check_%j.err

module load pytorch

cd /projappl/project_2019649/grain_research/code/grain_project
python scripts/csc_gpu_data_test.py
