#!/bin/bash
#SBATCH -p glinda
#SBATCH --job-name=gpfi
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/mfholth/slurm_output_python/logs/%x_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1


MODES=(standard extreme)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR=/home/mfholth/slurm_output_python/logs

PROJ_DIR="/home/mfholth/subseasonal/weekly_data/final_code"
export PYTHONPATH="${PROJ_DIR}/src:${PYTHONPATH}"

#### Ensure the output directory exists
mkdir -p $OUTPUT_DIR

source activate base
conda init
conda activate Subseasonal

python "${PROJ_DIR}/scripts/run_grouped_PFI.py" --mode $MODE

