"""RevIN-PatchTST domain FM, self-pretrained 20K steps on 50 Buildings-900K partitions. checkpoint RevIN_PatchTST_Foundation_v2_Mature.pth, 43 keys, 542K params. model def recovered from the training notebook so it stays key-compatible with the checkpoint, don't rename tensors. trained config: 512->96, patch 16 stride 8, d_model 64, 4 heads, 3 layers, dropout 0.1, MSE on the RevIN-normalised scale. historical fine-tune was full-param AdamW lr 5e-4, 8 epochs, sliding windows stride 4."""

from __future__ import annotations
import torch
import torch.nn as nn


class RevIN(nn.Module):
    # reversible instance norm, per-window mean/std with affine

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return x * self.stdev + self.mean


class PatchTST(nn.Module):
    # RevIN-PatchTST as trained, 512 h context -> 96 h horizon

    def __init__(self, context_len: int = 512, pred_len: int = 96,
                 patch_len: int = 16, stride: int = 8, d_model: int = 64,
                 n_heads: int = 4, e_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = int((context_len - patch_len) / stride + 1)

        self.revin = RevIN(num_features=1)
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.patch_num, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers)

        self.flatten = nn.Flatten(start_dim=-2)
        self.projection_head = nn.Linear(self.patch_num * d_model, pred_len)

    def forward(self, x):
        # x: (B, context_len, 1) raw kWh
        x = self.revin(x, "norm")
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x = x.squeeze(2)                                  # (B, patch_num, patch_len)
        x = self.value_embedding(x) + self.position_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.projection_head(x)                       # (B, pred_len)
        x = x.unsqueeze(-1)                               # (B, pred_len, 1)
        return self.revin(x, "denorm")                    # raw kWh


def load_patchtst_900k(checkpoint_path: str,
                       device: str | None = None,
                       eval_mode: bool = True) -> PatchTST:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTST()
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)                # strict: all 43 keys must match, no partial load
    n = sum(p.numel() for p in model.parameters())
    print(f"[P900K] Loaded {len(sd)} keys strict=True, {n/1e3:.0f}K params, "
          f"device={dev}")
    model = model.to(dev)
    if eval_mode:
        model.eval()
    return model
