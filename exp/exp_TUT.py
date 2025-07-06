import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from loguru import logger
import numpy as np
from tqdm import tqdm
import re
import os
from sklearn.metrics import balanced_accuracy_score
from models_v1.TUT import TUT
import wandb

from data_provider.data_factor import data_provider

from eval import segment_bars_with_confidence
from utils import KL_loss, SKL_loss, JS_loss, W_loss, L2_loss, CE_loss, class2boundary, extract_dis_from_attention, create_distribution_from_cls, plot_attention_map
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import matplotlib
import numpy as np
from matplotlib import font_manager

class ExpTUT:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.num_classes = configs.num_classes
        self.model = self._build_model(configs.model).to(configs.device)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        # logger.add('logs/' + args.dataset + "_" + args.split + "_{time}.log")
        # logger.add(sys.stdout, colorize=True, format="{message}")
    
    def _build_model(self, model_type):

        model = TUT(self.configs)

        return model

    def _get_data(self, mode):
        data_set, data_loader = data_provider(self.configs, mode)
        return data_set, data_loader

    def train(self):
        train_data, train_loader = self._get_data(mode='train')
        _, test_loader = self._get_data(mode='val')

        optimizer = optim.Adam(self.model.parameters(), lr=self.configs.lr, betas=(0.9, self.configs.adambeta), weight_decay=self.configs.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        self.model.train()

        best_balance_acc = 0.0
        patience = 5
        wait = 0
        for epoch in range(self.configs.num_epochs):
            # Log hyperparameters and start of run in WandB
            if getattr(self.configs, "use_wandb", False):
                wandb.config.update(vars(self.configs))
            epoch_loss = 0
            epoch_ba_loss = 0
            correct = 0
            total = 0
            for i, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
                batch_input = batch_data['feature'].to(self.device)
                batch_target = batch_data['label'].to(self.device)
                mask = batch_data['mask'].to(self.device)
                # Keep original targets for accuracy computation
                batch_target_orig = batch_target
                mask_orig = mask
               

                optimizer.zero_grad()
                predictions, all_attns = self.model(batch_input, mask)

                loss = torch.tensor(0.0).to(self.device)

                for p in predictions:
                    # Compute loss using temporal cropping without mutating originals
                    min_len = min(p.size(2), batch_target.size(1))
                    p_cut = p[:, :, :min_len]
                    target_cut = batch_target[:, :min_len]
                    mask_cut = mask[:, :min_len]

                    loss += self.ce(
                        p_cut.transpose(2, 1).contiguous().view(-1, self.num_classes),
                        target_cut.view(-1)
                    )
                    loss += self.configs.gamma * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p_cut[:, :, 1:], dim=1),
                                F.log_softmax(p_cut.detach()[:, :, :-1], dim=1)
                            ),
                            min=0, max=16
                        ) * mask_cut.unsqueeze(1)[..., 1:].expand_as(p_cut[:, :, 1:])
                    )

                if self.configs.baloss:
                    baloss = torch.tensor(0.0).to(self.device)
                    use_chi = False
                    loss_layer_num = 1
                    #    (1,L)      (begin_length)   (end_length)
                    # extract from all layers (different resolution) to get begin_index and end_index
                    _, begin_index, end_index = class2boundary(batch_target)
                    down_target = batch_target
                    begin_index_list = [begin_index]
                    end_index_list = [end_index]
                    B, _, L = batch_input.shape
                    # print(L)
                    len_list = [L // (2 ** i) for i in range(loss_layer_num + 1)]  # [L, L/2, L//4, ...]
                    for i in range(loss_layer_num):
                        down_target = F.interpolate(down_target.float().unsqueeze(0), size=len_list[i+1]).squeeze(0).long()
                        _, begin_index, end_index = class2boundary(down_target)
                        begin_index_list.append(begin_index)
                        end_index_list.append(end_index)

                    for attn in all_attns:  # each attn is each stage list
                        # attn: a list of (B, H, L, window_size)
                        for i in range(loss_layer_num):
                            # print(begin_index_list[i+1])
                            if begin_index_list[i+1].shape[0] > 0 and end_index_list[i+1].shape[0] > 0:
                                attn_begin = torch.index_select(attn[i], dim=2, index=begin_index_list[i+1].to(self.device))  # (B,H,l,window_size), encoder layer attn begin
                                attn_end = torch.index_select(attn[i], dim=2, index=end_index_list[i+1].to(self.device))  # (B,H,l,window_size), encoder layer attn end
                                baloss += self.configs.beta * KL_loss(attn_begin, create_distribution_from_cls(0, self.configs.window_size, use_chi).to(self.device))
                                baloss += self.configs.beta * KL_loss(attn_end, create_distribution_from_cls(2, self.configs.window_size, use_chi).to(self.device))
                            # print(attn_begin)
                            # print(attn_end)
                            # print(baloss)

                            # attn_begin = torch.index_select(attn[-i-1], dim=2, index=begin_index_list[i].to(device))  # (1,H,l,window_size), decoder layer attn begin
                            # attn_end = torch.index_select(attn[-i-1], dim=2, index=end_index_list[i].to(device))  # (1,H,l,window_size), decoder layer attn begin
                            # baloss += self.configs.beta * KL_loss(attn_begin, create_distribution_from_cls(0, self.configs.window_size, use_chi).to(device))
                            # baloss += self.configs.beta * KL_loss(attn_end, create_distribution_from_cls(2, self.configs.window_size, use_chi).to(device))
                        # break # comment on 50Salads and GTEA, meaning use all stages; if not comment, meaning only use prediction stage
                    epoch_ba_loss += baloss.item()
                    loss += baloss
        
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Compute accuracy using original targets cropped to last prediction length
                _, predicted = torch.max(predictions[-1], dim=1)
                min_len_last = min(predicted.size(1), batch_target_orig.size(1))
                predicted = predicted[:, :min_len_last]
                target_last = batch_target_orig[:, :min_len_last]
                mask_last = mask_orig[:, :min_len_last]
                correct += ((predicted == target_last).float() * mask_last).sum().item()
                total += mask_last.sum().item()

            scheduler.step(epoch_loss)
            # Log training metrics to WandB
            if getattr(self.configs, "use_wandb", False):
                wandb.log({
                    "train/loss": epoch_loss / len(train_data),
                    "train/accuracy": correct / total,
                    "epoch": epoch + 1
                }, step=epoch + 1)
            
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            # if self.configs.baloss:
            #     logger.info("[epoch %d]: epoch loss = %f, ba loss = %f, acc = %f" % (epoch + 1, epoch_loss / len(train_data), epoch_ba_loss / len(train_data),
            #                                                          float(correct)/total))
            # else:
            #     logger.info("[epoch %d]: epoch loss = %f, acc = %f" % (epoch + 1, epoch_loss / len(train_data),
            #                                                          float(correct)/total))

            # Early stopping based on balanced accuracy
            if (epoch + 1) % 5 == 0:
                val_bal_acc = self.test(test_loader, epoch)
                if val_bal_acc > best_balance_acc:
                    best_balance_acc = val_bal_acc
                    wait = 0
                    torch.save(self.model.state_dict(), self.configs.model_dir + "/best_epoch-" + str(epoch + 1) + ".model")
                    # Log validation metrics to WandB
                    if getattr(self.configs, "use_wandb", False):
                        wandb.log({
                            "val/balanced_accuracy": val_bal_acc,
                            "best/balanced_accuracy": best_balance_acc
                        }, step=epoch + 1)
                    logger.info(f"[EarlyStop] New best balanced_acc {best_balance_acc:.4f} at epoch {epoch+1}, saving model.")
                else:
                    wait += 1
                    logger.info(f"[EarlyStop] No improvement in balanced_acc at epoch {epoch+1} (wait {wait}/{patience}).")
                    # Log validation metrics to WandB even if not improved
                    if getattr(self.configs, "use_wandb", False):
                        wandb.log({
                            "val/balanced_accuracy": val_bal_acc,
                            "best/balanced_accuracy": best_balance_acc
                        }, step=epoch + 1)
                if wait >= patience:
                    logger.info(f"[EarlyStop] Stopping training at epoch {epoch+1} due to no improvement.")
                    break



    def test(self, test_loader, epoch):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        if_wrap = False
        with torch.no_grad():
            for i, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
                batch_input = batch_data['feature'].to(self.device)
                batch_target = batch_data['label'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                predictions, _ = self.model(batch_input, mask)
                # Frame-level logits for final stage
                logits = predictions[-1]  # shape [B, C, T_pred]
                _, predicted = torch.max(logits, dim=1)  # shape [B, T_pred]
                # Crop predictions, targets, and mask to minimum shared length
                min_len = min(predicted.size(1), batch_target.size(1))
                pred_cut = predicted[:, :min_len]
                target_cut = batch_target[:, :min_len]
                mask_cut = mask[:, :min_len]
                # Flatten batch and time dimensions
                pred_flat = pred_cut.contiguous().view(-1)
                target_flat = target_cut.contiguous().view(-1)
                mask_flat = mask_cut.contiguous().view(-1)
                # Select valid (non-zero mask) positions
                valid_positions = mask_flat > 0
                valid_preds = pred_flat[valid_positions]
                valid_targets = target_flat[valid_positions]
                # Collect for metrics
                all_preds.extend(valid_preds.detach().cpu().tolist())
                all_labels.extend(valid_targets.detach().cpu().tolist())
                # Compute frame-level accuracy
                correct += (valid_preds == valid_targets).sum().item()
                total += valid_positions.sum().item()

        acc = float(correct) / total
        balance_acc = balanced_accuracy_score(all_labels, all_preds)
        # Log evaluation metrics to WandB if enabled
        if getattr(self.configs, "use_wandb", False):
            wandb.log({
                "eval/accuracy": acc,
                "eval/balanced_accuracy": balance_acc
            }, step=epoch + 1)
        print(f"---[epoch {epoch+1}]---: test acc = { acc:.4f} ---: balance acc = {balance_acc:.4f} used acc this time")
        self.model.train()
        # test_loader.reset()
        return acc


    def find_latest_epoch(self, model_dir):
        if not os.path.exists(model_dir):
            return None
        epoch_files = [f for f in os.listdir(model_dir) if re.match(r'best_epoch-(\d+)\.model$',f)]
        if not epoch_files:
            return None
        return max([int(re.findall(r'\d+',f)[0]) for f in epoch_files])

    def save_horizontal_comparison(self, pred, gt, confidence,
                                name, save_path, actions_dict):

        # ---------- 字体 ----------
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
        FONT = {"fontname": font_name, "fontweight": "bold", "fontsize": 18}

        # ---------- 数据 ----------
        pred       = np.asarray(pred).squeeze().tolist()
        gt         = np.asarray(gt).squeeze().tolist()
        confidence = np.asarray(confidence).squeeze().tolist()

        # ---------- 画布 ----------
        fig, ax = plt.subplots(
            3, 1, figsize=(18, 6), sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 0.5]}
        )
        # 给 legend 预留 20% 宽度
        fig.subplots_adjust(left=0.05, right=0.80, top=0.90, bottom=0.10, hspace=0.25)

        # ---------- 颜色 ----------
        base = plt.get_cmap("Blues")
        palette = [base(0.2 + 0.6*i/11) for i in range(12)]  # values from 0.2 to 0.8
        label_colors = {i: palette[i] for i in range(12)}

        # ---------- Pred / GT 条纹 ----------
        for i, row in enumerate([pred, gt]):
            for t, lab in enumerate(row):
                ax[i].axvline(t, color=label_colors[lab], linewidth=8)
            ax[i].set_yticks([])
            ax[i].set_ylabel("Pred" if i == 0 else "GT", **FONT)
            ax[i].tick_params(axis="x", labelsize=12)

        # ---------- Confidence ----------
        ax[2].plot(confidence, color="blue", linewidth=1)
        ax[2].set_ylabel("Confidence", **FONT)
        ax[2].set_xlabel("Time (frames)", **FONT)
        ax[2].set_ylim(0, 1.0)
        ax[2].tick_params(labelsize=12)

        # ---------- 坐标刻度加粗 ----------
        for axis in ax:
            for lbl in axis.get_xticklabels() + axis.get_yticklabels():
                lbl.set_fontproperties(font_manager.FontProperties(fname=font_path))
                lbl.set_weight("bold")

        # ---------- 标题 ----------
        fig.suptitle(f"TUT Prediction vs Ground Truth: {name}", **FONT)

        # ---------- Legend ----------
        legend_handles = [
            mpatches.Patch(color=color, label=lab)
            for lab, idx in actions_dict.items()
            for k, color in label_colors.items() if k == idx
        ]

        # 用 fig.legend ‑‑ 放在 figure 坐标 (0.82,0.5) 处，垂直居中
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.82, 0.65),   # (x,y) in figure fraction
            borderaxespad=0.0,
            ncol=1,
            fontsize=12,
            frameon=True,
            edgecolor="black"
        )

        # ---------- 保存 ----------
        save_file = os.path.join(save_path, f"{name}_comparison.pdf")
        plt.savefig(save_file, format="pdf", bbox_inches="tight")
        plt.close()

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        import os, time
        import numpy as np
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, average_precision_score
        from eval import segment_bars_with_confidence
        

        self.model.eval()
        self.model.to(self.device)
        # load the specified epoch
        latest_model = self.find_latest_epoch(model_dir)
        print(f'used model {latest_model}')
        ckpt = os.path.join(model_dir, f"best_epoch-{latest_model}.model")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        # first pass: generate predictions, confidence maps, and save recognition files
        t0 = time.time()
        for batch_data in batch_gen_tst:
            batch_target = batch_data['label'].to(self.device)
            mask = batch_data['mask'].to(self.device)
            vids = batch_data['vid']
            vid = vids[0]
            # load precomputed features
            feat_base = os.path.join(features_path, vid.split('.')[0])
            features = None
            for ext in ('.npy', '.pth'):
                if os.path.exists(feat_base + ext):
                    features = np.load(feat_base + ext) if ext == '.npy' else torch.load(feat_base + ext).numpy()
                    break
            if features is None:
                raise FileNotFoundError(f"No feature file for {vid}")
            if features.shape[0] > features.shape[1]:
                features = features.T
            features = features[:, ::sample_rate]

            x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(self.device)
            # 下采样 mask 以与 features 对齐，而不是全部置 1
            mask = batch_data['mask'].to(self.device)[:, ::sample_rate]
            predictions, _ = self.model(x, mask)

            for stage_idx, out in enumerate(predictions):
                conf, pred = torch.max(F.softmax(out, dim=1), dim=1)
                conf = conf.squeeze().cpu().tolist()
                pred = pred.squeeze().cpu().tolist()
                tgt = batch_target[:, ::sample_rate].squeeze().cpu().tolist()
                # ---- Align prediction / target / confidence lengths ----
                min_len = min(len(pred), len(tgt), len(conf))
                pred = pred[:min_len]
                tgt  = tgt[:min_len]
                conf = conf[:min_len]
                segment_bars_with_confidence(os.path.join(results_dir, f"{vid}_stage{stage_idx}.png"),
                                             conf, tgt, pred)

                if stage_idx == len(predictions) - 1:
                    self.save_horizontal_comparison(pred=pred, gt=tgt, confidence=conf,
                                               name=vid, save_path=results_dir, actions_dict=actions_dict)

            # ---- Write frame‑level labels, aligned to GT length ----
            target_len = batch_target.size(1)          # original (unpadded) frame length
            rec = []
            keys = list(actions_dict.keys())
            vals = list(actions_dict.values())
            for label in pred:
                rec.extend([keys[vals.index(label)]] * sample_rate)

            # truncate or pad so that len(rec) == target_len
            if len(rec) > target_len:
                rec = rec[:target_len]
            elif len(rec) < target_len and len(rec) > 0:
                rec.extend([rec[-1]] * (target_len - len(rec)))

            fout = os.path.join(results_dir, vid.split('.')[0] + ".txt")
            with open(fout, "w") as f:
                f.write("### Frame level recognition: ###\n")
                f.write(" ".join(rec))

        t1 = time.time()

        # second pass: aggregate all preds and gts for metrics
        all_preds, all_gts = [], []
        for batch_data in batch_gen_tst:
            batch_target = batch_data['label']
            mask = batch_data['mask']
            mask = mask[:, ::sample_rate]           # 下采样使掩码与特征长度一致
            vids = batch_data['vid']
            vid = vids[0]
            feat_base = os.path.join(features_path, vid.split('.')[0])
            features = None
            for ext in ('.npy', '.pth'):
                if os.path.exists(feat_base + ext):
                    features = np.load(feat_base + ext) if ext == '.npy' else torch.load(feat_base + ext).numpy()
                    break
            if features.shape[0] > features.shape[1]:
                features = features.T
            features = features[:, ::sample_rate]

            x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(self.device)
            predictions, _ = self.model(x, mask)
            _, pred = torch.max(F.softmax(predictions[-1], dim=1), dim=1)
            pred = pred.squeeze().cpu().numpy()
            gt   = batch_target.squeeze().cpu().numpy()
            gt   = gt[::sample_rate]  # 下采样 ground‑truth 与预测序列对齐

            # --- Align lengths ---
            min_len = min(len(pred), len(gt), mask.shape[1])
            pred = pred[:min_len]
            gt   = gt[:min_len]
            mask_cut = mask.squeeze().cpu().numpy()[:min_len]

            # --- Select valid (non‑padded) positions, consistent with train/val ---
            valid_positions = mask_cut > 0
            pred_valid = pred[valid_positions]
            gt_valid   = gt[valid_positions]

            all_preds.extend(pred_valid.tolist())
            all_gts.extend(gt_valid.tolist())

        # ----- Debug: warn if predictions contain unseen classes -----
        pred_classes  = set(all_preds)
        true_classes  = set(all_gts)
        unseen_preds  = sorted(pred_classes - true_classes)
        if unseen_preds:
            print(f"⚠️  y_pred contains classes not present in y_true: {unseen_preds}")

        # compute metrics
        acc   = accuracy_score(all_gts, all_preds)
        f1    = f1_score(all_gts, all_preds, average='macro')
        bal   = balanced_accuracy_score(all_gts, all_preds)
        # mAP
        K = len(actions_dict)
        gt_oh = np.eye(K)[all_gts]
        pred_oh = np.eye(K)[all_preds]
        mAP = average_precision_score(gt_oh, pred_oh, average='macro')
        # segment-level F1
        def extract_segments(seq):
            segs, start = [], 0
            cur = seq[0]
            for i, v in enumerate(seq[1:], 1):
                if v != cur:
                    segs.append((cur, start, i-1))
                    cur, start = v, i
            segs.append((cur, start, len(seq)-1))
            return segs
        def seg_iou(a,b):
            s1,e1 = a; s2,e2 = b
            inter = max(0, min(e1,e2)-max(s1,s2)+1)
            union = (e1-s1+1)+(e2-s2+1)-inter
            return inter/union if union>0 else 0
        def seg_f1(y,t,iou):
            G = extract_segments(t); P = extract_segments(y)
            gtm,predm = {}, {}
            for lab,s,e in G:   gtm.setdefault(lab,[]).append((s,e))
            for lab,s,e in P:   predm.setdefault(lab,[]).append((s,e))
            TP=FP=FN=0
            for lab,ps in predm.items():
                for pseg in ps:
                    if max((seg_iou(pseg,gs) for gs in gtm.get(lab,[])), default=0) >= iou: TP+=1
                    else: FP+=1
            for lab,gs_list in gtm.items():
                for gseg in gs_list:
                    if max((seg_iou(gseg,ps) for ps in predm.get(lab,[])), default=0) < iou: FN+=1
            prec = TP/(TP+FP) if TP+FP>0 else 0
            rec  = TP/(TP+FN) if TP+FN>0 else 0
            return 2*prec*rec/(prec+rec) if prec+rec>0 else 0
        f1_seg = {thr: seg_f1(all_preds, all_gts, thr) for thr in (0.1,0.25,0.5)}
        # edit
        def lev_norm(p,t):
            def dedup(x):
                y=[x[0]]
                for v in x[1:]:
                    if v!=y[-1]: y.append(v)
                return y
            p,t = dedup(p), dedup(t)
            n,m = len(p), len(t)
            if n==0: return 0.0
            D = np.zeros((n+1,m+1),int)
            for i in range(n+1): D[i][0]=i
            for j in range(m+1): D[0][j]=j
            for i in range(1,n+1):
                for j in range(1,m+1):
                    c = 0 if p[i-1]==t[j-1] else 1
                    D[i][j] = min(D[i-1][j]+1,D[i][j-1]+1,D[i-1][j-1]+c)
            return 1 - D[n][m]/max(n,m)
        edit_score = lev_norm(all_preds, all_gts)

        # print metrics
        t_elapsed = t1 - t0
        print(f"✔️ Prediction Finished in {t_elapsed:.2f}s")
        print(f"📊 Frame-level Accuracy:          {acc:.4f}")
        print(f"📈 Frame-level F1 (macro):        {f1:.4f}")
        print(f"🔄 Frame-level Balanced-Acc:      {bal:.4f}")
        print(f"⭐ Frame-level mAP:               {mAP:.4f}")
        for thr,val in f1_seg.items():
            print(f"🎯 Segment-level F1 @ IoU {int(thr*100)}%:   {val:.4f}")
        print(f"✏️ Segment-level Edit Score:      {edit_score:.4f}")

        metrics = {"frame_acc":acc,"frame_f1":f1,"frame_balanced_acc":bal,"frame_mAP":mAP,"edit_score":edit_score}
        for thr,val in f1_seg.items():
            metrics[f"segment_f1_iou_{int(thr*100)}"] = val
        return metrics


if __name__ == '__main__':
    pass
        # Finish WandB run if used
