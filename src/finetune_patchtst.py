"""P3 fine-tuning engine for the domain FM. reproduces the historical recipe from PROTOCOL.md: full-param fine-tune, AdamW lr 5e-4 (wd 1e-5, matching pretrain), 8 fixed epochs, 512->96 windows at stride 4, MSE on raw kWh, batch 32, seed 7, no early stopping. anything off this recipe is a new arm, not a reproduction. slices come from pre-test hours only so no window ever touches the test period."""

from __future__ import annotations
import numpy as np
import torch

from src.patchtst_900k import PatchTST, load_patchtst_900k

RECIPE = dict(lr=5e-4, weight_decay=1e-5, epochs=8, stride=4,
              batch_size=32, grad_clip=1.0, seed=7)

SLICES = ("first_15", "last_15", "last_30", "full")


def slice_bounds(n_pretest: int, slice_name: str) -> tuple[int, int]:
    # [start, end) hour indices of the slice inside the pre-test region
    if slice_name == "first_15":
        return 0, int(round(0.15 * n_pretest))
    if slice_name == "last_15":
        return n_pretest - int(round(0.15 * n_pretest)), n_pretest
    if slice_name == "last_30":
        return n_pretest - int(round(0.30 * n_pretest)), n_pretest
    if slice_name == "full":
        return 0, n_pretest
    raise ValueError(f"unknown slice {slice_name!r}; valid: {SLICES}")


def _make_windows(seg: np.ndarray, ctx: int, pred: int, stride: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    span = ctx + pred
    if len(seg) < span:
        raise ValueError(f"slice too short for one window: {len(seg)} < {span}")
    starts = np.arange(0, len(seg) - span + 1, stride)
    X = np.stack([seg[s:s + ctx] for s in starts])
    Y = np.stack([seg[s + ctx:s + span] for s in starts])
    return (torch.tensor(X, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(Y, dtype=torch.float32).unsqueeze(-1))


def finetune_on_slice(checkpoint_path: str,
                      x: np.ndarray,
                      slice_name: str,
                      test_hours: int = 672,
                      end_hour: int | None = None,
                      device: str | None = None,
                      recipe: dict = RECIPE,
                      verbose: bool = False) -> PatchTST:
    # fresh copy from the checkpoint, fine-tuned on one slice. x is the full 2017 hourly series, slices from [0, end_hour - test_hours) never the test data
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(recipe["seed"])
    np.random.seed(recipe["seed"])

    end = len(x) if end_hour is None else end_hour
    test_start = end - test_hours
    pre = np.asarray(x[:test_start], dtype=np.float64)
    a, b = slice_bounds(len(pre), slice_name)
    seg = pre[a:b]

    model = load_patchtst_900k(checkpoint_path, device=dev, eval_mode=False)
    X, Y = _make_windows(seg, model.context_len, model.pred_len,
                         recipe["stride"])
    n = len(X)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                            weight_decay=recipe["weight_decay"])
    lossf = torch.nn.MSELoss()

    model.train()
    order = np.arange(n)
    trace = []
    for ep in range(recipe["epochs"]):
        np.random.shuffle(order)
        tot = 0.0
        for i in range(0, n, recipe["batch_size"]):
            idx = order[i:i + recipe["batch_size"]]
            xb = X[idx].to(dev)
            yb = Y[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           recipe["grad_clip"])
            opt.step()
            tot += loss.item() * len(idx)
        trace.append(round(tot / n, 6))
        if verbose:
            print(f"    [FT] {slice_name} epoch {ep+1}/{recipe['epochs']} "
                  f"mse={trace[-1]:.4f} ({n} windows)")
    model.eval()
    return model, trace


class FTAdapter:
    # harness-compatible adapter around a fine-tuned or base model

    def __init__(self, model: PatchTST, name: str, batch_size: int = 256):
        self.model = model
        self.name = name
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    def predict(self, contexts: np.ndarray, horizon: int):
        from experiments.zeroshot_eval import ForecastResult   # lazy, avoids a cycle
        x = contexts[:, -self.model.context_len:]
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                t = torch.tensor(x[i:i + self.batch_size],
                                 dtype=torch.float32,
                                 device=self.device).unsqueeze(-1)
                o = self.model(t)
                outs.append(o[:, :horizon, 0].float().cpu().numpy())
        return ForecastResult(point=np.concatenate(outs, axis=0))
