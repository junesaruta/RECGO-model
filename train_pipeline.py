import os
import re
import pandas as pd
from tqdm import trange, tnrange
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert4rec_dataset import Bert4RecDataset
from bert4rec_model import RecommendationModel, RecommendationTransformer
from rich.table import Column, Table
from rich import box
from rich.console import Console
from torch import cuda
from train_validate import train_step, validate_step
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import random
import numpy as np

device = T.device('cuda') if cuda.is_available() else T.device('cpu')


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================
# Early Stopping
# =====================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.bad_count = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            self.bad_count = 0
            return False, True  # (should_stop, improved)

        improved = False
        if self.mode == "min":
            if value < self.best - self.min_delta:
                improved = True
        else:  # "max"
            if value > self.best + self.min_delta:
                improved = True

        if improved:
            self.best = value
            self.bad_count = 0
            return False, True
        else:
            self.bad_count += 1
            should_stop = self.bad_count >= self.patience
            return should_stop, False


def trainer(
    data_params,
    model_params,
    loggers,
    optimizer_params=None,
    warmup_steps=False,
    output_dir="./models/",
    modify_last_fc=False,
    validation=5,
    exp_logger=None
):
    console = loggers.get("CONSOLE")
    train_logger = loggers.get("TRAIN_LOGGER")
    valid_logger = loggers.get("VALID_LOGGER")

    # ---------- output dir ----------
    if not os.path.exists(output_dir):
        console.log("OUTPUT DIRECTORY DOES NOT EXIST. CREATING...")
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "model_files"))
        os.mkdir(os.path.join(output_dir, "model_files_initial"))
    else:
        console.log("OUTPUT DIRECTORY EXISTS. CHECKING CHILD DIRECTORY...")
        if not os.path.exists(os.path.join(output_dir, "model_files")):
            os.mkdir(os.path.join(output_dir, "model_files"))
        if not os.path.exists(os.path.join(output_dir, "model_files_initial")):
            os.mkdir(os.path.join(output_dir, "model_files_initial"))

    # ---------- seed ----------
    console.log("SEED WITH: ", model_params.get("SEED"))
    T.manual_seed(model_params["SEED"])
    T.cuda.manual_seed(model_params["SEED"])
    np.random.seed(model_params.get("SEED"))
    random.seed(model_params.get("SEED"))
    T.backends.cudnn.deterministic = True

    # ---------- model ----------
    console.log("MODEL PARAMS: ", model_params)
    model = RecommendationTransformer(
        vocab_size=model_params.get("VOCAB_SIZE"),
        heads=model_params.get("heads", 1),
        layers=model_params.get("layers", 2),
        emb_dim=model_params.get("emb_dim", 16),
        pad_id=model_params.get("pad_id", 0),
        num_pos=model_params.get("history", 40),
    )
    console.log("MODEL output vocab (rec.out_features): ", model.rec.out_features)

    if model_params.get("trained"):
        console.log("TRAINED MODEL AVAILABLE. LOADING...")
        model.load_state_dict(T.load(model_params.get("trained"))["state_dict"])
        console.log("MODEL LOADED")

    if modify_last_fc:
        new_word_embedding = nn.Embedding(
            model_params.get("NEW_VOCAB_SIZE"),
            model_params.get("emb_dim"),
            0
        )
        new_word_embedding.weight.requires_grad = False
        new_word_embedding.weight[:model.encoder.word_embedding.weight.size(0)] = \
            model.encoder.word_embedding.weight.clone().detach()
        model.encoder.word_embedding = new_word_embedding
        model.encoder.word_embedding.weight.requires_grad = True

        new_lin_layer = nn.Linear(model_params.get("emb_dim"),
                                  model_params.get("NEW_VOCAB_SIZE"))
        new_lin_layer.weight.requires_grad = False
        new_lin_layer.weight[:model.lin_op.weight.size(0)] = \
            model.lin_op.weight.clone().detach()
        model.lin_op = new_lin_layer
        model.lin_op.weight.requires_grad = True

        console.log("MODEL LIN OP OUT FEATURES: ", model.lin_op.out_features)

    model = model.to(device)
    console.log(model)
    console.log(f"TOTAL NUMBER OF MODEL PARAMETERS: {round(count_model_parameters(model)/1e6, 2)} Million")

    # ---------- optimizer ----------
    optim_name = optimizer_params.get("OPTIM_NAME") if optimizer_params else None
    if optim_name == "SGD":
        optimizer = T.optim.SGD(params=model.parameters(), **optimizer_params.get("PARAMS"))
    elif optim_name == "ADAM":
        optimizer = T.optim.Adam(params=model.parameters(), **optimizer_params.get("PARAMS"))
    else:
        optimizer = T.optim.SGD(
            params=model.parameters(),
            lr=model_params.get("LEARNING_RATE"),
            momentum=0.8,
            nesterov=True
        )

    console.log("OPTIMIZER AND MODEL DONE")

    # Create early stopper (monitor valid_loss)
    # early_stopper = EarlyStopping(
    #     patience=10,
    #     min_delta=1e-2,
    #     mode="min"
    # )
    early_stopper = EarlyStopping(
        patience=50,
        min_delta=0.1,
        mode="max"
    )
    # =========================================================
    # DATA SPLIT: Leave-One-Out (USER LEVEL) + VALID HAS HISTORY
    # =========================================================
    console.log("CONFIGURING DATASET AND DATALOADER")
    console.log("DATA PARAMETERS: ", data_params)

    data = pd.read_csv(data_params.get("path"))
    user_col = data_params.get("group_by_col")
    item_col = data_params.get("data_col")
    time_col = data_params.get("timestamp_col")

    data = data.sort_values([user_col, time_col]).reset_index(drop=True)

    train_parts = []
    valid_last_rows = []

    for uid, g in data.groupby(user_col):
        g = g.sort_values(time_col)
        if len(g) < 2:
            continue
        valid_last_rows.append(g.iloc[-1])
        train_parts.append(g.iloc[:-1])

    train_data = pd.concat(train_parts, axis=0).reset_index(drop=True)
    valid_last = pd.DataFrame(valid_last_rows).reset_index(drop=True)
    valid_source = pd.concat([train_data, valid_last], axis=0).reset_index(drop=True)

    console.log("LEN OF TRAIN DATASET: ", len(train_data))
    console.log("LEN OF VALID SOURCE (train+last): ", len(valid_source))
    console.log("N USERS TRAIN: ", train_data[user_col].nunique())
    console.log("N USERS VALID (last rows): ", valid_last[user_col].nunique())

    # ---------- datasets ----------
    train_dataset = Bert4RecDataset(
        train_data,
        data_params.get("group_by_col"),
        data_params.get("data_col"),
        data_params.get("train_history", 38),
        data_params.get("valid_history", 38),
        data_params.get("padding_mode", "left"),
        "train",
        data_params.get("timestamp_col")
    )
    train_dl = DataLoader(train_dataset, **data_params.get("LOADERS").get("TRAIN"))

    valid_dataset = Bert4RecDataset(
        valid_source,
        data_params.get("group_by_col"),
        data_params.get("data_col"),
        data_params.get("train_history", 39),
        data_params.get("valid_history", 39),
        data_params.get("padding_mode", "left"),
        "valid",
        data_params.get("timestamp_col")
    )
    valid_dl = DataLoader(valid_dataset, **data_params.get("LOADERS").get("VALID"))

    console.save_text(os.path.join(output_dir, "logs_model_initialization.txt"), clear=True)

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    losses = []
    for epoch in tnrange(1, model_params.get("EPOCHS") + 1):
        if epoch % 3 == 0:
            clear_output(wait=True)

        train_loss, train_acc = train_step(
            model, device, train_dl, optimizer, warmup_steps,
            data_params.get("MASK"),
            model_params.get("CLIP"),
            data_params.get("chunkify")
        )

        train_logger.add_row(str(epoch), str(train_loss), str(train_acc))
        console.log(train_logger)

        # save initial model
        if epoch == 1:
            console.log("Saving Initial Model")
            T.save(model, os.path.join(output_dir, "model_files_initial", model_params.get("SAVE_NAME")))
            T.save(
                dict(
                    state_dict=model.state_dict(),
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    optimizer_dict=optimizer._optimizer.state_dict() if warmup_steps else optimizer.state_dict(),
                ),
                os.path.join(output_dir, "model_files_initial", model_params.get("SAVE_STATE_DICT_NAME"))
            )

        # save best (train loss)
        if epoch > 1 and (len(losses) == 0 or min(losses) > train_loss):
            console.log("SAVING BEST MODEL AT EPOCH -> ", epoch)
            console.log("LOSS OF BEST MODEL: ", train_loss)
            console.log("ACCURACY OF BEST MODEL: ", train_acc)
            T.save(model, os.path.join(output_dir, "model_files", model_params.get("SAVE_NAME")))
            T.save(
                dict(
                    state_dict=model.state_dict(),
                    epoch=epoch,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    optimizer_dict=optimizer._optimizer.state_dict() if warmup_steps else optimizer.state_dict(),
                ),
                os.path.join(output_dir, "model_files", model_params.get("SAVE_STATE_DICT_NAME"))
            )

        losses.append(train_loss)

        # ---------- validation + early stop ----------
        if validation and epoch > 1 and epoch % validation == 0:
            valid_loss, metrics, total = validate_step(
                model, valid_dl, device,
                Ks=(1, 5, 10),
                PAD=data_params.get("PAD", 0),
                MASK=data_params.get("MASK", 1)
            )

            hit10  = metrics["Recall@1"]
            r5  = metrics["Recall@5"]
            r10 = metrics["Recall@10"]

            n1  = metrics["NDCG@1"]
            n5  = metrics["NDCG@5"]
            ndcg10 = metrics["NDCG@10"]

            m10 = metrics["MRR@10"]

            valid_logger.add_row(
                str(epoch),
                f"{valid_loss:.4f}",
                f"{hit10:.2f}",
                f"{r5:.2f}",
                f"{r10:.2f}",
                f"{n1:.2f}",
                f"{n5:.2f}",
                f"{ndcg10:.2f}",
                f"{m10:.2f}",
                str(total)
            )

            console.log(valid_logger)
            console.log("VALIDATION DONE AT EPOCH ", epoch)
            # Early stopping check (use ndcg10)
            should_stop, improved = early_stopper.step(ndcg10)

            if improved:
                console.log(f"EarlyStop: improved NDCG@10 -> {ndcg10:.4f} (loss={valid_loss:.4f})")
                T.save(
                    dict(
                        state_dict=model.state_dict(),
                        epoch=epoch,
                        valid_loss=float(valid_loss),

                        recall1=float(metrics["Recall@1"]),
                        recall5=float(metrics["Recall@5"]),
                        recall10=float(metrics["Recall@10"]),

                        ndcg1=float(metrics["NDCG@1"]),
                        ndcg5=float(metrics["NDCG@5"]),
                        ndcg10=float(metrics["NDCG@10"]),

                        mrr10=float(metrics["MRR@10"]),  # ปกติใช้ @10 ตัวเดียว

                        optimizer_dict=optimizer._optimizer.state_dict() if warmup_steps else optimizer.state_dict(),
                    ),
                    os.path.join(output_dir, "model_files", "best_by_ndcg.pth")
                )

            else:
                console.log(f"EarlyStop: no improvement ({early_stopper.bad_count}/{early_stopper.patience}) "
                            f"(best_ndcg={early_stopper.best:.4f}, current_ndcg={ndcg10:.4f})")

            if should_stop:
                console.log(f"Early stopping at epoch {epoch}. Best NDCG@10 = {early_stopper.best:.4f}")
                break

        console.save_text(os.path.join(output_dir, "logs_training.txt"), clear=True)

