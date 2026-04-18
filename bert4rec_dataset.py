import torch as T
from torch.utils.data import Dataset
import random
import pandas as pd
from typing import List, Tuple

from constants import TRAIN_CONSTANTS


class Bert4RecDataset(Dataset):
    """
    Stable Bert4RecDataset with optional chunkify/windowing.

    Key features:
    - chunkify=True: create multiple windows per user (index_map), reduces randomness/variance.
    - deterministic padding (no random left/right each call): uses padding_mode from init.
    - train split: windows end before last valid_history items (consistent with original logic).
    """

    def __init__(
        self,
        data_csv: pd.DataFrame,
        group_by_col: str,
        data_col: str,
        train_history: int = 38,
        valid_history: int = 39,
        padding_mode: str = "left",
        split_mode: str = "train",
        timestamp_col: str = "Date",
        # ---- new knobs ----
        chunkify: bool = False,
        stride: int = None,          # default = train_history
        min_seq_len: int = 2,        # minimum usable length before padding
        random_end: bool = False,    # if chunkify=False, optionally randomize end_ix (old behavior-ish)
        seed: int = None,            # optional: make index_map stable across runs
    ):
        super().__init__()

        self.data_csv = data_csv
        self.group_by_col = group_by_col
        self.data_col = data_col
        self.train_history = int(train_history)
        self.valid_history = int(valid_history)
        self.padding_mode = padding_mode
        self.split_mode = split_mode
        self.timestamp_col = timestamp_col

        self.pad = TRAIN_CONSTANTS.PAD
        self.mask = TRAIN_CONSTANTS.MASK

        self.chunkify = bool(chunkify)
        self.stride = int(stride) if stride is not None else int(train_history)
        self.min_seq_len = int(min_seq_len)
        self.random_end = bool(random_end)

        if seed is not None:
            random.seed(seed)

        # group
        self.groups_df = self.data_csv.groupby(by=self.group_by_col)
        self.groups = list(self.groups_df.groups)

        # build index map: list[(group_key, end_ix)]
        self.index_map: List[Tuple[object, int]] = []
        self._build_index_map()

    # ----------------------------
    # helpers
    # ----------------------------
    def pad_sequence(self, tokens: List[int]) -> List[int]:
        """Pad tokens to train_history using self.padding_mode (deterministic)."""
        if len(tokens) < self.train_history:
            pad_len = self.train_history - len(tokens)
            if self.padding_mode == "left":
                tokens = [self.pad] * pad_len + tokens
            else:
                tokens = tokens + [self.pad] * pad_len
        return tokens

    def mask_sequence(self, sequence: List[int], p_keep: float = 0.8) -> List[int]:
        """
        Randomly mask tokens: keep original with prob p_keep, else replace with MASK.
        """
        out = []
        for s in sequence:
            out.append(s if random.random() < p_keep else self.mask)
        return out

    def mask_last_elements_sequence(self, sequence: List[int], p_keep: float = 0.5) -> List[int]:
        """
        Only mask the last valid_history items.
        """
        if self.valid_history <= 0:
            return sequence
        if len(sequence) <= self.valid_history:
            # if too short, mask whatever exists
            return self.mask_sequence(sequence, p_keep=p_keep)

        head = sequence[:-self.valid_history]
        tail = sequence[-self.valid_history:]
        return head + self.mask_sequence(tail, p_keep=p_keep)

    def _build_index_map(self):
        """
        Build index_map to stabilize sampling.

        - train:
          max_end = n - valid_history  (same spirit as original)
          if chunkify: end_ix moves by stride (plus final max_end)
          else: one end_ix per user (max_end or random)
        - valid/test:
          end_ix = n (one window per user)
        """
        for gkey in self.groups:
            gdf = (
                self.groups_df.get_group(gkey)
                .sort_values(by=[self.timestamp_col])
                .reset_index(drop=True)
            )
            n = len(gdf)

            if n < self.min_seq_len:
                continue

            if self.split_mode == "train":
                max_end = n - self.valid_history
                if max_end < self.min_seq_len:
                    continue

                if self.chunkify:
                    # create multiple windows
                    end = self.min_seq_len
                    while end <= max_end:
                        self.index_map.append((gkey, end))
                        end += self.stride

                    # ensure we include max_end as last window (important)
                    if len(self.index_map) == 0 or self.index_map[-1][0] != gkey or self.index_map[-1][1] != max_end:
                        self.index_map.append((gkey, max_end))
                else:
                    if self.random_end:
                        # old-ish behavior: random end_ix in [min_seq_len, max_end]
                        end_ix = random.randint(self.min_seq_len, max_end)
                    else:
                        end_ix = max_end
                    self.index_map.append((gkey, end_ix))

            elif self.split_mode in ["valid", "test"]:
                # one window per user at the end
                self.index_map.append((gkey, n))
            else:
                raise ValueError(
                    f"Split should be either of `train`, `valid`, or `test`. {self.split_mode} is not supported"
                )

    # ----------------------------
    # core
    # ----------------------------
    def _get_window_items(self, group_df: pd.DataFrame, end_ix: int) -> List[int]:
        start_ix = max(0, end_ix - self.train_history)
        seq = group_df.iloc[start_ix:end_ix]
        items = seq[self.data_col].tolist()
        return items

    def get_item(self, idx: int):
        gkey, end_ix = self.index_map[idx]

        group_df = (
            self.groups_df.get_group(gkey)
            .sort_values(by=[self.timestamp_col])
            .reset_index(drop=True)
        )

        trg_items = self._get_window_items(group_df, end_ix)

        # safety: if empty
        if len(trg_items) == 0:
            trg_items = [self.pad]  # fallback minimal

        # build source
        if self.split_mode == "train":
            src_items = self.mask_sequence(trg_items, p_keep=0.8)
        else:
            src_items = self.mask_last_elements_sequence(trg_items, p_keep=0.5)

        # deterministic padding (no randomness)
        trg_items = self.pad_sequence(trg_items)
        src_items = self.pad_sequence(src_items)

        trg_mask = [1 if t != self.pad else 0 for t in trg_items]
        src_mask = [1 if t != self.pad else 0 for t in src_items]

        return {
            "source": T.tensor(src_items, dtype=T.long),
            "target": T.tensor(trg_items, dtype=T.long),
            "source_mask": T.tensor(src_mask, dtype=T.long),
            "target_mask": T.tensor(trg_mask, dtype=T.long),
        }

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index: int):
        return self.get_item(index)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data = pd.read_csv("user_mapped66-1.csv")

    ds = Bert4RecDataset(
        data_csv=data,
        group_by_col="UserID",
        data_col="itemId_mapped",
        train_history=45,
        valid_history=46,
        padding_mode="left",
        split_mode="train",
        timestamp_col="Date",
        chunkify=True,
        stride=45,        # ลอง 45 ก่อน (นิ่งสุด) แล้วค่อยลอง 20
        random_end=False,
        seed=42,
    )

    dl = DataLoader(ds, batch_size=32, shuffle=True)
    batch = next(iter(dl))
    print({k: v.shape for k, v in batch.items()})
