import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder


class RecModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 heads=2,
                 layers=1,
                 emb_dim=16,
                 pad_id=0,
                 num_pos=16):
        print("vocab_siz=",vocab_size)
        """
        Args:
            vocab_size (int): จำนวน token ทั้งหมดใน vocabulary
            heads (int): จำนวน attention heads (multi-head self-attention)
            layers (int): จำนวน encoder layers
            emb_dim (int): ขนาดของ embedding vector
            pad_id (int): token id ที่ใช้เป็น padding
            num_pos (int): maximum sequence length (สำหรับ positional encoding)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.pad_id = pad_id
        self.num_pos = num_pos
        self.vocab_size = vocab_size
        self.channel_dim = num_pos * emb_dim
        self.encoder = Encoder(source_vocab_size=vocab_size,
                               emb_dim=emb_dim,
                               layers=layers,
                               heads=heads,
                               dim_key=emb_dim,
                               dim_value=emb_dim,
                               dim_model=emb_dim,
                               dim_inner=emb_dim * 2,
                               pad_id=pad_id,
                               num_pos=num_pos)
        self.lin_op = nn.Linear(emb_dim, vocab_size) #inear projection weight: [32,1400]

    def forward(self, batch):
        # batch: [B, T] ของ item ids อยู่แล้ว
        x = self.encoder(batch, None)     # [B, T, emb_dim]
        x = self.lin_op(x)                # [B, T, vocab_size]
        return x
    
class RecommendationModel(nn.Module):
    """Sequential recommendation model architecture
    """
    def __init__(self,
                 vocab_size,
                 heads=2,
                 layers=1,
                 emb_dim=16,
                 pad_id=0,
                 num_pos=16,
                 num_channels=16):
        """Recommendation model initializer

        Args:
            vocab_size (int): Number of unique tokens/items
            heads (int, optional): Number of heads in the Multi-Head Self Attention Transformers (). Defaults to 4.
            layers (int, optional): Number of Layers. Defaults to 6.
            emb_dim (int, optional): Embedding Dimension. Defaults to 512.
            pad_id (int, optional): Token used to pad tensors. Defaults to 0.
            num_pos (int, optional): Positional Embedding, fixed sequence. Defaults to 120.
            num_channels (int, optional): Channels for Item Embeddings
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.pad_id = pad_id
        self.num_pos = num_pos
        self.vocab_size = vocab_size
        print(f'NUM POS: ', num_pos)
        self.item_embeddings = nn.Embedding(self.vocab_size,
                                            embedding_dim=emb_dim)
        self.encoder = Encoder(
            source_vocab_size=num_channels,
            emb_dim=num_channels,
            layers=layers,
            heads=heads,
            dim_key=num_channels,
            dim_value=num_channels,
            dim_model=num_channels,
            dim_inner=num_channels * 4,
            pad_id=pad_id,
            num_pos=num_pos,
        )
        self.lin_op = nn.Linear(num_channels, self.vocab_size)

    def forward(self, batch):
        """Returns predictions for a given sequence of tokens/items

        Args:
            batch (torch.Tensor): Batch of sequences

        Returns:
            torch.Tensor: Prediction for masked items
        """
        src_items = self.item_embeddings(batch)
        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        batch = T.arange(0, in_sequence_len,
                         device=src_items.device).unsqueeze(0).repeat(
                             batch_size, 1)
        op = self.encoder(batch, None)
        op = op.permute(1, 0, 2)
        op = self.lin_op(op)
        return op
    
class RecommendationTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 heads=2,
                 layers=1,
                 emb_dim=32,
                 pad_id=0,
                 num_pos=100):
        super().__init__()
        self.emb_dim = emb_dim
        self.pad_id = pad_id
        self.num_pos = num_pos
        self.vocab_size = vocab_size

        self.encoder = Encoder(
            source_vocab_size=vocab_size,
            emb_dim=emb_dim,
            layers=layers,
            heads=heads,
            dim_model=emb_dim,
            dim_inner=emb_dim,
            dim_value=emb_dim,
            dim_key=emb_dim,
            pad_id=self.pad_id,
            num_pos=num_pos
        )

        # projection -> vocab
        self.rec = nn.Linear(emb_dim, vocab_size, bias=False)

        # weight tying (ต้องขนาดตรงกัน: [vocab_size, emb_dim])
        self.rec.weight = self.encoder.word_embedding.weight

    def make_pad_mask(self, source):
        """
        source: [B, T]
        return mask: [B, 1, 1, T] (รูปแบบนิยมใช้กับ attention)
        ถ้า Encoder ของจูนรับ mask คนละรูปแบบ ให้ปรับตรงนี้
        """
        pad_mask = (source == self.pad_id)              # [B, T] True ที่เป็น pad
        return pad_mask.unsqueeze(1).unsqueeze(2)       # [B,1,1,T]

    def forward(self, source, source_mask=None):
        """
        source: [B, T]
        return: logits [B, T, V]
        """
        if source_mask is None:
            source_mask = self.make_pad_mask(source)

        enc_op = self.encoder(source, source_mask)      # [B, T, emb]
        logits = self.rec(enc_op)                       # [B, T, V]
        return logits
