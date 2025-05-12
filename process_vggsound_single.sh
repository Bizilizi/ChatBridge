#!/bin/sh
#SBATCH --job-name="chatbridge"
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zverev@in.tum.de
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --error=./logs/slurm-%A_%a.out
 
nvidia-smi

# Activate your conda environment (adjust if needed)
set -x

modality=$1
echo "This is $modality, page $SLURM_ARRAY_TASK_ID"

# Set the appropriate prompt based on the modality
if [ "$modality" = "a" ]; then
    PROMPT="Classes: {cl}. From the given list of classes, which ones do you hear in this audio? Answer using the exact names of the classes, separated by commas."
else
    PROMPT="Classes: {cl}. From the given list of classes, which ones do you see or hear in this video? Answer using the exact names of the classes, separated by commas."
fi

# Run the script on each node, assigning each task to a different GPU
srun python process_vggsound.py \
  --cfg-path eval_configs/chatbridge_eval.yaml \
  --gpu-id 0 \
  --dataset_path /mnt/lustre/work/akata/askoepke97/data/vggsound \
  --video_csv ../../data/test.csv \
  --output_csv csv/$modality/predictions.csv \
  --temperature 1.0 \
  --num_beams 1 \
  --max_new_tokens 300 \
  --max_length 2000 \
  --page $SLURM_ARRAY_TASK_ID \
  --per_page 1000 \
  --modality $modality \
  --prompt_mode single \
  --prompt "$PROMPT"