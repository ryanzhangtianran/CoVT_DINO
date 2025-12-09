#!/usr/bin/env bash
# 四卡4090烟雾测试脚本：固定全部参数
set -e

# 如需切换视频解码后端，可取消下一行注释并按需设为 pyav/decord/torchvision
# export LEROBOT_VIDEO_BACKEND=pyav

accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  --num_machines 1 \
  --dynamo_backend no \
  /home/zhongzd/trzhang/repos/CoVT_DINO/train/src/train_covt_dino.py \
  --smoke-test \
  --max-test-batches 2 \
  --test-batch-size 4 
