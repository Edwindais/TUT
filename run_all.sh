#!/usr/bin/env bash
set -e  # 任一命令失败即退出

# ======== 全局配置 =========
DATASET="surgery_I3D_new"
GPU_ID=0
# 五折 split
SPLITS=(1 2 3 4 5)
# 用于 eval.py 的 epoch （如果你在 predict 阶段用 --resume_model_path 指定）
EPOCH=200
ROOT_PATH=/mnt/hdd2/Projects_dai/Projects_dai/MS-TCN2_dai/data/

# ======== Step 1: Train 每个 split =========
# echo "🚀 开始训练 Split ${SPLITS[*]} （使用 GPU $GPU_ID）"
# for SPLIT in "${SPLITS[@]}"; do
#   echo -e "\n➡️ [TRAIN] Split $SPLIT"
#   CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
#     --model TUT \
#     --action train \
#     --num_epochs $EPOCH \
#     --dataset $DATASET \
#     --root_path $ROOT_PATH \
#     --split $SPLIT \
#     --use_wandb \
#     --num_workers 6 \
#     --wandb_project "TUT-Surgery" \
#     # --bz 4 \
# done

# #======== Step 2: Predict 每个 split =========
echo -e "\n🔮 开始对 Split ${SPLITS[*]} 进行预测"
for SPLIT in "${SPLITS[@]}"; do
  echo -e "\n➡️ [PREDICT] Split $SPLIT"
  CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --model TUT \
    --action predict \
    --dataset $DATASET \
    --root_path $ROOT_PATH \
    --split $SPLIT 
done

# # ======== Step 3: 评估整体平均性能（split=0）=========
# echo -e "\n📊 开始评估 split=0 （平均五折结果）"
# python eval.py \
#   --dataset $DATASET \
#   --split 0 \
#   --result_dir "./results/${DATASET}"

# echo -e "\n✅ 全部完成！"