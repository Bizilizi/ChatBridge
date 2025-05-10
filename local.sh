mkdir -p /tmp/vggsound
/usr/bin/squashfuse /dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/zverev/datasets/vggsound.squashfs /tmp/vggsound

python process_vggsound.py \
  --cfg-path eval_configs/chatbridge_eval.yaml \
  --gpu-id 0 \
  --dataset_path /tmp/vggsound \
  --video_csv ../../data/train.csv \
  --output_csv csv/v/predictions.csv \
  --temperature 1.0 \
  --num_beams 1 \
  --max_new_tokens 300 \
  --max_length 2000 \
  --page 1 \
  --per_page 10 \
  --modality v \
  --prompt_mode single \
  --prompt "Do you hear or see '{cl}' class in this video? Answer only with yes or no."