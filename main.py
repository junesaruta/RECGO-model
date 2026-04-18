# main2.py (NO popularity penalty)
import os
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table, Column
from rich import box
from train_pipeline import trainer

# ---------- Rich loggers ----------
console = Console(record=True)
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Loss", justify="center"),
    Column("Accuracy", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

valid_logger = Table(
    Column("Epoch", justify="center"),
    Column("Loss", justify="center"),
    Column("Recall@1", justify="center"),
    Column("Recall@5", justify="center"),
    Column("Recall@10", justify="center"),
    Column("NDCG@1", justify="center"),
    Column("NDCG@5", justify="center"),
    Column("NDCG@10", justify="center"),
    Column("MRR@10", justify="center"),
    Column("total", justify="center"),
    title="Validation Status",
    pad_edge=False,
    box=box.ASCII,
)

loggers = dict(CONSOLE=console, TRAIN_LOGGER=training_logger, VALID_LOGGER=valid_logger)

# ---------- Config ----------
data_params = dict(
    path="new-senior/bert4rec-main/scripts/user_mapped66-1.csv",
    group_by_col="UserID",
    data_col="itemId_mapped",
    train_history=45,
    valid_history=46,
    padding_mode="left",
    MASK=1,
    chunkify=False,
    threshold_column=None,
    threshold=None,
    timestamp_col="Date",
    LOADERS=dict(
        TRAIN=dict(batch_size=128, shuffle=True,  num_workers=0),
        VALID=dict(batch_size=128, shuffle=False, num_workers=0),
    ),
)

model_params = dict(
    SEED=42,
    VOCAB_SIZE=584,
    heads=4,
    layers=2,
    emb_dim=64,
    pad_id=0,
    history=46,
    trained=None,
    EPOCHS=500,
    dropout=0.3,
    SAVE_NAME="bert4rec.pt",
    SAVE_STATE_DICT_NAME="bert4rec-state-dict.pth",
    CLIP=1,
)

optimizer_params = {"OPTIM_NAME": "ADAM", "PARAMS": {"lr": 5e-3, "weight_decay": 1e-4}}
output_dir = "new-senior/models/recommendation-transformer"

PAD_ID = model_params["pad_id"]
MASK_ID = data_params["MASK"]
MAX_LEN = data_params["train_history"]
VOCAB_SIZE = model_params["VOCAB_SIZE"]

def make_input_sequence(items, max_len=MAX_LEN, pad_id=PAD_ID):
    items = list(items)[-max_len:]
    pad_len = max_len - len(items)
    return [pad_id] * pad_len + items

@torch.no_grad()
def recommend_topk_for_user(model, user_items, device, k=10, exclude_seen=True):
    """
    Recommend top-k items for a user by predicting the [MASK] position at the end.
    NO popularity penalty.
    """
    model.eval()

    # seq: last MAX_LEN-1 items + [MASK]
    seq = list(user_items)[-(MAX_LEN - 1):] + [MASK_ID]
    seq = make_input_sequence(seq, MAX_LEN, PAD_ID)

    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
    out = model(x)

    # รองรับทั้ง [B,T,V] และ [B,V,T]
    if out.dim() == 3:
        if out.shape[1] == MAX_LEN:
            logits = out[:, -1, :]      # [1,V]  (B,T,V)
        else:
            logits = out[:, :, -1]      # [1,V]  (B,V,T)
    else:
        logits = out                    # [1,V]

    scores = logits.squeeze(0).clone()  # [V]

    # กัน PAD/MASK
    if 0 <= PAD_ID < scores.numel():
        scores[PAD_ID] = -1e9
    if 0 <= MASK_ID < scores.numel():
        scores[MASK_ID] = -1e9

    # กันแนะนำของที่เคยซื้อแล้ว
    if exclude_seen:
        seen = set(int(i) for i in user_items if i != PAD_ID)
        seen.add(PAD_ID)
        seen.add(MASK_ID)
        for it in seen:
            if 0 <= it < scores.numel():
                scores[it] = -1e9

    topk = torch.topk(scores, k=k).indices.detach().cpu().tolist()
    return topk

def main():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "model_files"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "model_files_initial"), exist_ok=True)

    # ---------- TRAIN ----------
    trainer(
        data_params=data_params,
        model_params=model_params,
        loggers=loggers,
        warmup_steps=False,
        output_dir=output_dir,
        modify_last_fc=False,
        validation=1,
        optimizer_params=optimizer_params,
    )

    # ---------- Top-K Demo ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(data_params["path"])
    df[data_params["timestamp_col"]] = pd.to_datetime(df[data_params["timestamp_col"]])
    df = df.sort_values([data_params["group_by_col"], data_params["timestamp_col"]])

    def get_user_history(user_id):
        return df.loc[df[data_params["group_by_col"]] == user_id, data_params["data_col"]].astype(int).tolist()

    best_path = os.path.join(output_dir, "model_files", model_params["SAVE_STATE_DICT_NAME"])

    from bert4rec_model import RecommendationTransformer

    model = RecommendationTransformer(
        vocab_size=VOCAB_SIZE,
        heads=model_params["heads"],
        layers=model_params["layers"],
        emb_dim=model_params["emb_dim"],
        pad_id=model_params["pad_id"],
        num_pos=model_params["history"],
    ).to(device)

    # ---- Load checkpoint (safe) ----
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        # older torch that doesn't support weights_only
        ckpt = torch.load(best_path, map_location=device)

    # รองรับทั้งกรณี ckpt เป็น dict มี state_dict หรือเป็น state_dict ตรงๆ
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # # demo 3 users (fixed)
    # target_users = ["78487#11", "54003#1", "39017#1"]
    # sample_users = (
    #     df[data_params["group_by_col"]]
    #     .drop_duplicates()
    #     .loc[lambda x: x.isin(target_users)]
    #     .tolist()
    # )

    # for uid in sample_users:
    #     hist = get_user_history(uid)
    #     recs = recommend_topk_for_user(model, hist, device, k=10, exclude_seen=True)

    #     console.print(f"\n[bold]User:[/bold] {uid}")
    #     console.print(f"History last 10: {hist[-10:]}")
    #     console.print(f"[green]Top-10 Rec (no popularity penalty):[/green] {recs}")
    sample_users = (
    df[data_params["group_by_col"]]
    .drop_duplicates()
    .tolist()
    )

    results = []

    for uid in sample_users:
        hist = get_user_history(uid)
        if len(hist) == 0:
            continue

        recs = recommend_topk_for_user(model, hist, device, k=10, exclude_seen=True)

        # print
        console.print(f"\n[bold]User:[/bold] {uid}")
        console.print(f"History last 10: {hist[-10:]}")
        console.print(f"[green]Top-10 Rec (no popularity penalty):[/green] {recs}")

        # save result
        results.append({
            "UserID": uid,
            "History_last10": hist[-10:],
            "Top10_Rec": recs
        })

    # ---------- Save to CSV ----------
    out_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "all_user_recommendations.csv")
    out_df.to_csv(out_path, index=False)

    console.print(f"\n📁 Saved all recommendations to: [bold]{out_path}[/bold]")

if __name__ == "__main__":
    main()
