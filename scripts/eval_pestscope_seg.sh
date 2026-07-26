#!/bin/sh

# Evaluate PestScope from a DeepSpeed checkpoint.
# Put the checkpoint directory in --resume. With --eval_only, metrics/results are written to result.txt.
# For PDRES evaluation, remove --reason_seg_data_add and --reason_image_root_path.

deepspeed \
  --master_port=24969 \
  --include=localhost:1 \
  train_ft.py \
  --version="Model/GLaMM-RefSeg/" \
  --dataset_dir="refer_data/dif_data/" \
  --reason_seg_data_add="reason_data/dif_data" \
  --reason_image_root_path="images/" \
  --image_root_path="images/" \
  --vision_pretrained="Model/SAM-vit-H/sam_vit_h_4b8939.pth" \
  --vision-tower="Model/clip-vit-large-patch14-336/" \
  --exp_name="PestScope-7b" \
  --lora_r=8 \
  --lr=0.0001 \
  --pretrained \
  --use_segm_data \
  --seg_dataset="Refer_Segm" \
  --segm_sample_rates="1" \
  --refer_segm_data="refcocog" \
  --val_dataset="RefCOCOgSegmVal" \
  --epochs=10 \
  --steps_per_epoch=1208 \
  --mask_validation \
  --grad_accumulation_steps=2 \
  --batch_size=8 \
  --eval_only \
  --resume="/output/PestScope-7b/ckpt_model_5/"
