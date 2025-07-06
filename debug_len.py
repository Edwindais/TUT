#!/usr/bin/env python3
import argparse, os, torch, numpy as np
from exp.exp_TUT import ExpTUT
from data_provider.data_factor import data_provider

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='TUT')
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')

parser.add_argument("--root_path",
                    default="/mnt/hdd2/Projects_dai/Projects_dai/ASFormer/data/",
                    help="Root directory with <dataset>/features, groundtruth, splits, mapping.txt")
# hyper-parameter
parser.add_argument('--window_size', type=int, default=31, help='window size')
parser.add_argument('--l_seg', type=int, default=200, help='the length of segment')

parser.add_argument('--input_dim', type=int, default=1024, help='dim of input')

parser.add_argument('--d_model_PG', type=int, default=64, help='hidden dimension')
parser.add_argument('--d_ffn_PG', type=int, default=128, help='ffn dimension')
parser.add_argument('--n_heads_PG', type=int, default=4, help='heads num')

parser.add_argument('--d_model_R', type=int, default=32, help='hidden dimension')
parser.add_argument('--d_ffn_R', type=int, default=64, help='ffn dimension')
parser.add_argument('--n_heads_R', type=int, default=4, help='heads num')

parser.add_argument('--input_dropout', type=float, default=0.3, help='dropout2d rate')
parser.add_argument('--attention_dropout', type=float, default=0.2, help='dropout rate in attention')
parser.add_argument('--ffn_dropout', type=float, default=0.2, help='dropout rate in ffn')

parser.add_argument('--pre_norm', action="store_true", help='pre or post layernorm, default False')
parser.add_argument('--rpe_share', action="store_true", help='PG and R stage share rpe, default False')
parser.add_argument('--rpe_use', action="store_true", help='use rpe or not, default False')
parser.add_argument('--activation', type=str, default='relu', help='activation')

parser.add_argument('--num_layers_PG', type=int, default=5, help='prediction generation layer num')
parser.add_argument('--num_layers_R', type=int, default=5, help='refinement layer num')
parser.add_argument('--num_R', type=int, default=3, help='refinement stage num')

parser.add_argument('--gamma', type=float, default=0.15, help='tmse loss weight')
parser.add_argument('--beta', type=float, default=0.15, help='boundary-aware loss weight')
parser.add_argument('--baloss', action="store_true", help='use boundary-aware loss or not, default False')

# training-parameter
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--adambeta', type=float, default=0.98)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--bz', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--video_idx', type=int, default=-1,
                    help='index of batch/video to inspect; -1 = check ALL videos')

parser.add_argument('--train_ratio', type=float, default=1.0)
args = parser.parse_args()

# ---------- basic setup ----------
args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
dataset2num = {"surgery_I3D_new": 12, "gtea": 11, "50salads": 19, "breakfast": 48}
args.num_classes = dataset2num[args.dataset]
args.model_dir   = f"./checkpoints/{args.dataset}/split_{args.split}"
args.window_size = 31        #
args.l_seg       = 200
# 其余超参省略，默认即可
my_exp = ExpTUT(configs=args)

# ---------- load any checkpoint, or随机权重 ----------
ckpt_files = [
    f for f in os.listdir(args.model_dir)
    if f.startswith("best_epoch") and f.endswith(".model")
]
if not ckpt_files:
    raise FileNotFoundError(f"No checkpoint matching 'best_epoch-*.model' in {args.model_dir}")

# 选数字最大的 epoch 作为“最新”模型
latest_ckpt = max(ckpt_files, key=lambda s: int(s.split("-")[1].split(".")[0]))
ckpt_path   = os.path.join(args.model_dir, latest_ckpt)
my_exp.model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
print(f"✓ loaded {ckpt_path}")

# ---------- DataLoader ----------
data_set, data_loader = data_provider(args, mode="test")
sample_rate = data_set.__get_sample_rate__()
print(f"SAMPLE_RATE  = {sample_rate}")

# ---------- Inspect batches ----------
loader_list = list(data_loader)
if args.video_idx >= 0:
    loader_list = [loader_list[args.video_idx]]      # 只检查指定样本

mismatches = []   # 收集 (vid, feat_len, gt_len, mask_len, pred_len)

for idx, batch in enumerate(loader_list):
    # DataLoader 已在 batch 维度 stack；batch_size 默认为 1
    feat  = batch["feature"][0]    # (C, L_raw)
    label = batch["label"][0]      # (L_gt,)
    mask  = batch["mask"][0]       # (L_mask,)
    vid   = batch["vid"][0]

    raw_feat_len  = feat.shape[1]          # L_raw
    raw_gt_len    = label.shape[0]
    raw_mask_len  = mask.shape[0]

    # ----- same preprocessing as predict() -----
    features = feat.numpy()
    if features.shape[0] > features.shape[1]:
        features = features.T
    features = features[:, ::sample_rate]
    x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(args.device)  # (1, C, L')
    x_len = x.size(2)

    mask_pred = torch.ones((1, x_len), device=args.device)
    preds, _ = my_exp.model(x, mask_pred)
    pred_len = preds[-1].size(2)   # 取最后输出长度

    # ---- consistency check ----
    consistent = (raw_feat_len == raw_gt_len == raw_mask_len ==
                  x_len == pred_len)
    if not consistent:
        mismatches.append((vid, raw_feat_len, raw_gt_len,
                           raw_mask_len, pred_len))

    # 若只看单个样本，打印详细信息
    if args.video_idx >= 0:
        print(f"\n===== {vid} =====")
        print(f"feature_len={raw_feat_len}, gt_len={raw_gt_len}, "
              f"mask_len={raw_mask_len}, x_len={x_len}, pred_len={pred_len}")

# ---------- summary ----------
if args.video_idx < 0:
    if mismatches:
        print("\n❌  FOUND LENGTH MISMATCHES:")
        for v, lf, lg, lm, lp in mismatches:
            print(f"{v:>25} | feat:{lf:>6d}  gt:{lg:>6d}  mask:{lm:>6d}  pred:{lp:>6d}")
    else:
        print("\n✅  All videos have consistent lengths.")