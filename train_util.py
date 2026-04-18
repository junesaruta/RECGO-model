import torch as T
import torch.nn.functional as F
from typing import List

def ensure_B_T_V(logits, labels):
    # logits: 3D
    # labels: [B,T]
    B, T = labels.shape[0], labels.shape[1]
    if logits.dim() != 3:
        raise ValueError(f"Expected 3D logits, got {logits.shape}")

    # [B,T,V] → OK
    if logits.shape[0] == B and logits.shape[1] == T:
        return logits

    # [T,B,V] → permute to [B,T,V]
    if logits.shape[0] == T and logits.shape[1] == B:
        return logits.permute(1, 0, 2).contiguous()

    # [B,V,T] → permute to [B,T,V]
    if logits.shape[0] == B and logits.shape[2] == T:
        return logits.permute(0, 2, 1).contiguous()

    raise ValueError(f"Cannot infer layout for logits {logits.shape} vs labels {labels.shape}")

PAD, MASK = 0, 1

def calculate_loss(y_pred, y_true, mask):
    y_pred = ensure_B_T_V(y_pred, y_true)           # ← normalize to [B,T,V]
    pmask  = mask.bool() & (y_true != PAD)          # [B,T]
    if not pmask.any():
        return y_pred.new_tensor(0.0)
    logits_m = y_pred[pmask]                        # [M,V]
    labels_m = y_true[pmask]                        # [M]
    import torch.nn.functional as F
    return F.cross_entropy(logits_m, labels_m)

def calculate_accuracy(y_pred, y_true, mask):
    y_pred = ensure_B_T_V(y_pred, y_true)           # [B,T,V]
    pmask  = mask.bool() & (y_true != PAD)
    if not pmask.any():
        import torch as T
        return T.tensor(0.0, dtype=T.float64)
    pred = y_pred.argmax(dim=-1)                    # over V
    return (pred[pmask] == y_true[pmask]).double().mean()



def calculate_combined_mean(batch_sizes: List, means: List):
    """Combined Mean of batch accuracy

    Args:
        batch_sizes (List): Number of items in the batch at every iteration
        means (List): Accuracy of every batch prediction
    
    Returns:
        float: Epoch Accuracy
    """

    combined_mean = (T.sum(T.tensor(batch_sizes) * T.tensor(means)) /
                     T.sum(T.tensor(batch_sizes))) * 100
    return combined_mean