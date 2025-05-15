#!/bin/bash
#SBATCH --job-name="vjobid_1882js8"
#SBATCH --nodes=79
#SBATCH --ntasks=316
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_bprod
#SBATCH --mem=450G
#SBATCH --time=07:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a-sophia.koepke@uni-tuebingen.de
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.out
#SBATCH --exclude=lrdn[3018,0013,0021,0441,0176,0304,0353,0666,0815,1018,1086,1370,1384,1420,1444,1478,1488,1506,1509,1520,1542,1556,1569-1570,1580,1597,1607,1647,1656,1702,1718,1771,1779,1810,1835,1927,1930,1953,2000,2013,2028,2065,2101,2128,2136,2157,2179,2277,2285,2309,2337,2342,2344-2345,2348-2349,2354,2357,2363,2368,2371,2381,2384,2387,2418,2444,2465,2490,2562-2563,2585,2590,2595,2597-2598,2652,2654,2662,2674,2682,2711,2716,2731,2734,2736,2746,2751,2755,2772,2791-2792,2794,2820,2852,2912,2917,2921,2926-2927,2976,2984,2997,3004,3032,3058,3067,3072,3089,3110-3111,3123,3136,3145,3151,3161,3200,3221,3277,3395,3444]

nvidia-smi

# Activate your conda environment (adjust if needed)
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_OFFLINE=1
    
modality=$1
echo "This is $modality"

# Set the appropriate prompt based on the modality
PROMPT="Do you see or hear \"{cl}\" class in this video? Answer only with yes or no."


srun bash -c "python process_vggsound.py \
  --cfg-path eval_configs/chatbridge_eval.yaml \
  --gpu-id ${SLURM_LOCALID} \
  --dataset_path /leonardo_work/EUHPC_E03_068/akoepke/vs \
  --video_csv ../../data/test.csv \
  --output_csv csv/$modality/predictions.csv \
  --temperature 1.0 \
  --num_beams 1 \
  --max_new_tokens 100 \
  --max_length 2000 \
  --page \$SLURM_PROCID \
  --per_page 49 \
  --modality $modality \
  --prompt_mode multi \
  --prompt \"$PROMPT\"
"