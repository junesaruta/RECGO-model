from tqdm import tqdm, tqdm_notebook
from train_util import *
from torch.autograd import Variable
import torch as T


def train_step(model,
               device,
               loader,
               optimizer,
               scheduled_optim=False,
               MASK=1,
               CLIP=2,
               chunkify=False):
    """Train batch step

    Args:
        model: Model
        device (torch.device): Device to train on 'cuda'/'cpu'
        loader (torch.utils.data.dataloader.DataLoader): DataLoader object
        optimizer: Optimizer
        MASK (int, optional): Mask TOKEN ID. Defaults to 1.

    Returns:
        Tuple(float, float): Epoch/Step Loss, Epoch Accuracy
    """
    model.train()
    total_loss = 0
    total_counts = 0
    train_accs = []
    train_bs = []
    for _, batch in enumerate(tqdm_notebook(loader, desc="Train Loader")):

        if chunkify and _ % chunkify != 0:
            continue

        source = Variable(batch["source"].to(device))
        source_mask = Variable(batch["source_mask"].to(device))
        target = Variable(batch["target"].to(device))
        target_mask = Variable(batch["target_mask"].to(device))
        train_bs += [source.size(0)]
        # mask = target_mask == MASK
        # mask = (source == MASK)
        mask = (target_mask == MASK)

        optimizer.zero_grad()

        output = model(source, source_mask)

        loss = calculate_loss(output, target, mask)

        total_counts += 1
        total_loss += loss.item()

        loss.backward()
        if CLIP:
            T.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if scheduled_optim:
            optimizer.step_and_update_lr()
        else:
            optimizer.step()

        mean = calculate_accuracy(output, target, mask)
        train_accs += [mean.item()]

    epoch_acc = calculate_combined_mean(train_bs, train_accs)

    return total_loss / total_counts, epoch_acc.item()

def validate_step(
    model, loader, device,
    Ks=(1,5,10),
    PAD=0, MASK=1,
    print_hist=True,
    export_path=None
):
    import torch as T
    import torch.nn.functional as F
    import pandas as pd
    from tqdm import tqdm

    model.eval()

    maxK = max(Ks)

    loss_sum = 0.0
    total = 0

    hit_sums  = {k: 0.0 for k in Ks}
    ndcg_sums = {k: 0.0 for k in Ks}
    mrr_sums  = {k: 0.0 for k in Ks}

    rank_hist = T.zeros(maxK + 1, dtype=T.long)

    export_rows = []   # <<< EXPORT BUFFER
    shown_debug = False

    with T.no_grad():
        for batch in tqdm(loader, desc="Valid Loader"):
            source = batch["source"].to(device)           # [B,L]
            source_mask = batch["source_mask"].to(device) # [B,L]
            target = batch["target"].to(device)           # [B,L]

            B, L = source.shape

            # ---- find last valid index (robust padding) ----
            valid_users = source_mask.sum(dim=1) > 0
            if valid_users.sum().item() == 0:
                continue

            idx = T.arange(L, device=device).unsqueeze(0).expand(B, L)
            last_idx = (idx * source_mask.long()).max(dim=1).values.long()
            b_idx = T.arange(B, device=device)
            # ------------------------------------------------

            output = model(source, source_mask)

            # model output fix [B,V,L] -> [B,L,V]
            if output.dim() == 3 and output.shape[1] != source.shape[1]:
                output = output.permute(0, 2, 1)

            scores = output[b_idx, last_idx, :]   # [B,V]
            true_item = target[b_idx, last_idx]   # [B]

            # filter invalid target
            valid2 = valid_users & (true_item != PAD) & (true_item != MASK)
            if valid2.sum().item() == 0:
                continue

            scores = scores[valid2]
            true_item = true_item[valid2]

            V = scores.size(1)
            in_range = (true_item >= 0) & (true_item < V)
            if in_range.sum().item() == 0:
                if not shown_debug:
                    print("[VALID DEBUG] output:", output.shape)
                    print("[VALID DEBUG] scores:", scores.shape)
                    shown_debug = True
                continue

            scores = scores[in_range]
            true_item = true_item[in_range]
            N = true_item.size(0)
            if N == 0:
                continue

            # block PAD / MASK
            if PAD < V:  scores[:, PAD] = -1e9
            if MASK < V: scores[:, MASK] = -1e9

            # loss
            loss = F.cross_entropy(scores, true_item, reduction="mean")
            loss_sum += loss.item() * N
            total += N

            # ranking
            K_use = min(maxK, V)
            topk = T.topk(scores, k=K_use, dim=1).indices
            matches = (topk == true_item.unsqueeze(1))

            found_any = matches.any(dim=1)
            ranks = T.full((N,), -1, device=device, dtype=T.long)
            ranks[found_any] = matches[found_any].float().argmax(dim=1)

            # rank histogram
            r_cpu = ranks.detach().cpu()
            if (r_cpu >= 0).any():
                counts = T.bincount(r_cpu[r_cpu >= 0], minlength=maxK)
                rank_hist[:maxK] += counts[:maxK]
            rank_hist[maxK] += (r_cpu < 0).sum()

            ndcg_all = T.zeros(N, device=device)
            mrr_all  = T.zeros(N, device=device)
            ok = ranks >= 0
            ndcg_all[ok] = 1.0 / T.log2(ranks[ok].float() + 2.0)
            mrr_all[ok]  = 1.0 / (ranks[ok].float() + 1.0)

            for k in Ks:
                k_eff = min(k, K_use)
                hit_k = (ranks >= 0) & (ranks < k_eff)
                hit_sums[k]  += hit_k.float().sum().item()
                ndcg_sums[k] += (ndcg_all * hit_k.float()).sum().item()
                mrr_sums[k]  += (mrr_all  * hit_k.float()).sum().item()

            # ---------- EXPORT ----------
            if export_path is not None:
                topk_cpu = topk.detach().cpu()
                true_cpu = true_item.detach().cpu()
                ranks_cpu = ranks.detach().cpu()

                for i in range(N):
                    export_rows.append({
                        "true_item": int(true_cpu[i]),
                        "rank": int(ranks_cpu[i]),
                        "found": int(ranks_cpu[i] >= 0),
                        "topk_items": topk_cpu[i].tolist()
                    })
            # ----------------------------

    if total == 0:
        return 0.0, {}, 0

    epoch_loss = loss_sum / total

    metrics = {}
    for k in Ks:
        metrics[f"Recall@{k}"] = hit_sums[k] / total * 100.0
        metrics[f"NDCG@{k}"]   = ndcg_sums[k] / total * 100.0
        metrics[f"MRR@{k}"]    = mrr_sums[k] / total * 100.0

    if print_hist:
        print(f"\n[Rank Histogram] top{maxK} (total={total})")
        for r in range(maxK):
            cnt = int(rank_hist[r])
            print(f"  rank={r:2d}: {cnt:6d} ({cnt/total*100:5.2f}%)")
        nf = int(rank_hist[maxK])
        print(f"  not_found: {nf:6d} ({nf/total*100:5.2f}%)\n")

    if export_path is not None and len(export_rows) > 0:
        pd.DataFrame(export_rows).to_csv(export_path, index=False)
        print(f"[VALID] export -> {export_path}")

    return float(epoch_loss), metrics, int(total)