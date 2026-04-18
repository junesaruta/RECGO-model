from train_pipeline import trainer

from rich.table import Column, Table
from rich import box
from rich.console import Console

console = Console(record=True)

training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Loss", justify="center"),
    Column("Accuracy", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

valid_loggger = Table(
    Column("Epoch", justify="center"),
    Column("Loss", justify="center"),
    Column("Accuracy", justify="center"),
    title="Validation Status",
    pad_edge=False,
    box=box.ASCII,
)

loggers = dict(CONSOLE=console,
               TRAIN_LOGGER=training_logger,
               VALID_LOGGER=valid_loggger)

model_params = dict(
    SEED=2,
    VOCAB_SIZE=1425,
    heads=1,
    layers=2,
    emb_dim=32,
    pad_id=0,
    history=100,
    # trained=
    # "C:/Users/LENOVO/Desktop/new-senior/bert4rec-main/models/recommendation-transformer/model_files/bert4rec-state-dict.pth",
    # # trained=None,
    LEARNING_RATE=1e-2,
    EPOCHS=10,
    SAVE_NAME="bert4rec.pt",
    SAVE_STATE_DICT_NAME="bert4rec-state-dict.pth",
    CLIP=2

    # NEW_VOCAB_SIZE=59049
)

data_params = dict(
    # path="/content/bert4rec/data/ratings_mapped.csv",
    #  path="drive/MyDrive/bert4rec/data/ml-25m/ratings_mapped.csv",
    path="data1/user_mapped99.csv",
    group_by_col="UserID",
    data_col="itemId_mapped",
    train_history=100,
    valid_history=5,
    padding_mode="right",
    MASK=1,
    chunkify=False,
    timestamp_col="Date",
    LOADERS=dict(TRAIN=dict(batch_size=32, shuffle=True, num_workers=0),
                 VALID=dict(batch_size=32, shuffle=False, num_workers=0)))

optimizer_params = {
    "OPTIM_NAME": "ADAM",
    "PARAMS": {
        "lr": 1e-2,
    }
}
# optimizer_params = {
#     "OPTIM_NAME": "SGD",
#     "PARAMS": {
#         "lr": 0.142,
#         "momentum": 0.85,
#     }
# }

output_dir = "C:/Users/LENOVO/Desktop/new-senior/bert4rec-main/models/recommendation-transformer/rec-transformer-model-10/"

trainer(data_params=data_params,
        model_params=model_params,
        loggers=loggers,
        warmup_steps=False,
        output_dir=output_dir,
        modify_last_fc=False,
        validation=False,
        optimizer_params=optimizer_params)
