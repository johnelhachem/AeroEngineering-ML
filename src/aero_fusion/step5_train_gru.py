"""Train the GRU v2 trajectory reconstruction model (Colab-ready)."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DRIVE_ROOT = Path("/content/drive/MyDrive/aero_project")
DATA_DIR   = DRIVE_ROOT / "step4"
OUTPUT_DIR = DRIVE_ROOT / "step5"

HIDDEN_SIZE  = 160
NUM_LAYERS   = 2
DROPOUT      = 0.2
BATCH_SIZE   = 32
EPOCHS       = 60
LR           = 1e-3
LR_PATIENCE  = 8
EARLY_STOP   = 15
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
SEED         = 42

BEFORE_STEPS   = 64
AFTER_STEPS    = 32
N_SEQ_FEATURES = 6
MAX_ADSC_WP    = 32
EARTH_R        = 6_371_000.0

ALT_LOSS_WEIGHT = 0.001
FT_PER_M        = 3.28084
ALT_NORM_CENTER = 35000.0   # ft
ALT_NORM_SCALE  = 5000.0    # ft

N_LAST_DYN = N_SEQ_FEATURES

def haversine_m_np(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def haversine_loss(pred_lat, pred_lon, true_lat, true_lon, mask):
    """Exact haversine loss in metres."""
    lat1 = pred_lat * (math.pi / 180.0)
    lat2 = true_lat * (math.pi / 180.0)
    dlat = (true_lat - pred_lat) * (math.pi / 180.0)
    dlon = (true_lon - pred_lon) * (math.pi / 180.0)
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    dist = 2 * EARTH_R * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))
    return (dist * mask).sum() / mask.sum().clamp(min=1.0)

def altitude_loss(pred_alt, true_alt, mask, valid_alt_mask):
    combined_mask = mask * valid_alt_mask
    if combined_mask.sum() < 1:
        return torch.tensor(0.0)
    diff = (pred_alt - true_alt) * combined_mask
    return (diff**2).sum() / combined_mask.sum().clamp(min=1.0)

def gc_interpolate_batch(lat0, lon0, lat1, lon1, tau):
    lat0r = np.radians(lat0[:, None]); lon0r = np.radians(lon0[:, None])
    lat1r = np.radians(lat1[:, None]); lon1r = np.radians(lon1[:, None])
    x0 = np.cos(lat0r)*np.cos(lon0r); y0 = np.cos(lat0r)*np.sin(lon0r); z0 = np.sin(lat0r)
    x1 = np.cos(lat1r)*np.cos(lon1r); y1 = np.cos(lat1r)*np.sin(lon1r); z1 = np.sin(lat1r)
    dot = np.clip(x0*x1 + y0*y1 + z0*z1, -1, 1)
    omega = np.arccos(dot); sin_o = np.sin(omega)
    safe = (sin_o > 1e-10).astype(float); sos = np.where(sin_o > 1e-10, sin_o, 1.0)
    w0 = np.sin((1-tau)*omega)/sos*safe + (1-safe); w1 = np.sin(tau*omega)/sos*safe
    xp, yp, zp = w0*x0+w1*x1, w0*y0+w1*y1, w0*z0+w1*z1
    n = np.sqrt(xp**2 + yp**2 + zp**2).clip(min=1e-10)
    return np.degrees(np.arcsin(np.clip(zp/n, -1, 1))), np.degrees(np.arctan2(yp/n, xp/n))

class TrajectoryDataset(Dataset):
    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.before_seq   = d["before_seq"].astype(np.float32)
        self.before_mask  = d["before_mask"].astype(np.float32)
        self.after_seq    = d["after_seq"].astype(np.float32)
        self.after_mask   = d["after_mask"].astype(np.float32)
        self.adsc_targets = d["adsc_targets"].astype(np.float32)  # (N, K, 2) lat/lon
        self.adsc_tau     = d["adsc_tau"].astype(np.float32)
        self.adsc_mask    = d["adsc_mask"].astype(np.float32)
        self.gap_dur_sec  = d["gap_dur_sec"].astype(np.float32)

        bla = d["before_anchor_lat"].astype(np.float32)
        blo = d["before_anchor_lon"].astype(np.float32)
        ala = d["after_anchor_lat"].astype(np.float32)
        alo = d["after_anchor_lon"].astype(np.float32)
        bl_lat, bl_lon = gc_interpolate_batch(bla, blo, ala, alo, self.adsc_tau)
        self.baseline_lat = bl_lat.astype(np.float32)
        self.baseline_lon = bl_lon.astype(np.float32)
        self.gap_norm = (self.gap_dur_sec / 6000.0).astype(np.float32)

        N = len(self.before_seq)
        self.last_dyn = np.zeros((N, N_LAST_DYN), dtype=np.float32)
        for i in range(N):
            valid_steps = np.where(self.before_mask[i] > 0)[0]
            if len(valid_steps) > 0:
                self.last_dyn[i] = self.before_seq[i, valid_steps[-1], :]

        N, K = self.adsc_targets.shape[:2]
        self.alt_targets    = np.full((N, K), ALT_NORM_CENTER, dtype=np.float32)
        self.alt_valid_mask = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            valid_steps = np.where(self.before_mask[i] > 0)[0]
            if len(valid_steps) > 0:
                last_step = valid_steps[-1]
                alt_norm  = float(self.before_seq[i, last_step, 5])
                if abs(alt_norm) < 5:
                    alt_m  = alt_norm * 1000.0 + 10500.0
                    alt_ft = alt_m * FT_PER_M
                    self.alt_targets[i, :]    = alt_ft
                    self.alt_valid_mask[i, :] = 1.0

        print(f"  {path.name}: {len(self)} samples  "
              f"(alt valid: {int(self.alt_valid_mask[:,0].sum())}/{N})")

    def __len__(self):
        return len(self.before_seq)

    def __getitem__(self, i):
        return dict(
            before_seq   = self.before_seq[i],
            before_mask  = self.before_mask[i],
            after_seq    = self.after_seq[i],
            after_mask   = self.after_mask[i],
            adsc_tau     = self.adsc_tau[i],
            adsc_mask    = self.adsc_mask[i],
            true_lat     = self.adsc_targets[i, :, 0],
            true_lon     = self.adsc_targets[i, :, 1],
            true_alt_ft  = self.alt_targets[i],
            alt_valid    = self.alt_valid_mask[i],
            baseline_lat = self.baseline_lat[i],
            baseline_lon = self.baseline_lon[i],
            gap_norm     = self.gap_norm[i],
            last_dyn     = self.last_dyn[i],
        )

class TrajectoryGRUv2(nn.Module):
    """Run the bidirectional-encoder GRU reconstruction model."""
    def __init__(self, D=N_SEQ_FEATURES, H=HIDDEN_SIZE, L=NUM_LAYERS, p=DROPOUT):
        super().__init__()

        self.before_enc = nn.GRU(D, H, L, batch_first=True, bidirectional=True,
                                  dropout=p if L > 1 else 0.)
        self.after_enc  = nn.GRU(D, H, L, batch_first=True, bidirectional=False,
                                  dropout=p if L > 1 else 0.)

        C = 2*H + H + 1 + N_LAST_DYN
        self.ctx_to_h0 = nn.Linear(C, H)
        self.decoder_gru = nn.GRU(3, H, 1, batch_first=True)
        self.decoder_head = nn.Sequential(
            nn.Linear(H, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self._init()

    def _init(self):
        for n, p in self.named_parameters():
            if "weight_ih" in n:    nn.init.xavier_uniform_(p)
            elif "weight_hh" in n:  nn.init.orthogonal_(p)
            elif "bias" in n:       nn.init.zeros_(p)
            elif "weight" in n and p.dim() == 2: nn.init.xavier_uniform_(p)

    def _enc(self, enc, seq, mask):
        lengths = mask.sum(1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths, batch_first=True, enforce_sorted=False)
        _, h = enc(packed)
        return h

    def forward(self, b):
        hb = self._enc(self.before_enc, b["before_seq"], b["before_mask"])
        ha = self._enc(self.after_enc,  b["after_seq"],  b["after_mask"])

        ctx = torch.cat([
            hb[-2], hb[-1],
            ha[-1],
            b["gap_norm"].unsqueeze(-1),
            b["last_dyn"],
        ], dim=-1)

        h0 = torch.tanh(self.ctx_to_h0(ctx)).unsqueeze(0)

        dec_in = torch.stack([
            b["adsc_tau"],
            b["baseline_lat"],
            b["baseline_lon"],
        ], dim=-1)

        gru_out, _ = self.decoder_gru(dec_in, h0)
        res = self.decoder_head(gru_out)

        pred_lat = b["baseline_lat"] + res[:, :, 0]
        pred_lon = b["baseline_lon"] + res[:, :, 1]
        pred_alt = ALT_NORM_CENTER   + res[:, :, 2] * ALT_NORM_SCALE

        return pred_lat, pred_lon, pred_alt

def to_dev(batch, dev):
    return {k: v.to(dev) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}

def run_epoch(model, loader, opt, dev, train):
    model.train(train)
    tot_loss = tot_pos = tot_alt = n = 0
    with torch.set_grad_enabled(train):
        for b in loader:
            b = to_dev(b, dev)
            pl, ql, al = model(b)
            pos_loss    = haversine_loss(pl, ql, b["true_lat"], b["true_lon"], b["adsc_mask"])
            alt_loss_val = altitude_loss(al, b["true_alt_ft"], b["adsc_mask"], b["alt_valid"])
            loss = pos_loss + ALT_LOSS_WEIGHT * alt_loss_val
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            with torch.no_grad():
                errs = haversine_m_np(
                    pl.detach().cpu().numpy(), ql.detach().cpu().numpy(),
                    b["true_lat"].cpu().numpy(), b["true_lon"].cpu().numpy())
                vm = b["adsc_mask"].cpu().numpy() > 0
                if vm.sum() > 0:
                    tot_pos += errs[vm].mean()
                    av = b["alt_valid"].cpu().numpy() > 0
                    if av.sum() > 0:
                        alt_diff = np.abs(al.detach().cpu().numpy() -
                                          b["true_alt_ft"].cpu().numpy())
                        tot_alt += alt_diff[av].mean()
            tot_loss += loss.item(); n += 1
    return tot_loss / max(n, 1), tot_pos / max(n, 1), tot_alt / max(n, 1)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    print("\nLoading data...")
    tr = TrajectoryDataset(DATA_DIR / "sequences_train.npz")
    va = TrajectoryDataset(DATA_DIR / "sequences_val.npz")
    te = TrajectoryDataset(DATA_DIR / "sequences_test.npz")
    trl = DataLoader(tr, BATCH_SIZE, shuffle=True,  num_workers=0)
    val = DataLoader(va, BATCH_SIZE, shuffle=False, num_workers=0)
    tel = DataLoader(te, BATCH_SIZE, shuffle=False, num_workers=0)

    model = TrajectoryGRUv2().to(dev)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, LR_PATIENCE)

    print(f"\n{'Epoch':>6}  {'TrLoss':>10}  {'TrPos':>10}  {'TrAlt':>10}  "
          f"{'VaLoss':>10}  {'VaPos':>10}")
    print("-" * 70)

    history = []; best_val = float("inf"); best_ep = 0; no_imp = 0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, tp, ta = run_epoch(model, trl, opt, dev, True)
        vl, vp, va_ = run_epoch(model, val, opt, dev, False)
        sch.step(vl)
        print(f"{ep:>6}  {tl:>10.1f}  {tp/1000:>9.2f}km  {ta:>9.0f}ft  "
              f"{vl:>10.1f}  {vp/1000:>9.2f}km  ({time.time()-t0:.0f}s)")
        history.append({
            "epoch": ep, "train_loss": tl, "train_pos_m": tp, "train_alt_ft": ta,
            "val_loss": vl, "val_pos_m": vp, "val_alt_ft": va_,
            "lr": opt.param_groups[0]["lr"],
        })
        if vl < best_val:
            best_val = vl; best_ep = ep; no_imp = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP:
                print(f"Early stop at epoch {ep}, best={best_ep}"); break

    json.dump(
        [{k: float(v) if hasattr(v, "item") else v for k, v in h.items()} for h in history],
        open(OUTPUT_DIR / "training_history.json", "w"), indent=2)

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", map_location=dev))
    model.eval()

    PLs, PLo, PAs, TLs, TLo, TAs, BLs, BLo, Ms = [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for b in tel:
            bd = to_dev(b, dev); pl, po, pa = model(bd)
            PLs.append(pl.cpu().numpy()); PLo.append(po.cpu().numpy())
            PAs.append(pa.cpu().numpy())
            TLs.append(b["true_lat"].numpy()); TLo.append(b["true_lon"].numpy())
            TAs.append(b["true_alt_ft"].numpy())
            BLs.append(b["baseline_lat"].numpy()); BLo.append(b["baseline_lon"].numpy())
            Ms.append(b["adsc_mask"].numpy())

    PL = np.concatenate(PLs); PO = np.concatenate(PLo); PA = np.concatenate(PAs)
    TL = np.concatenate(TLs); TO = np.concatenate(TLo); TA = np.concatenate(TAs)
    BL = np.concatenate(BLs); BO = np.concatenate(BLo)
    M  = np.concatenate(Ms)

    ge = haversine_m_np(PL, PO, TL, TO)
    be = haversine_m_np(BL, BO, TL, TO)
    gru_pf = np.array([ge[i][M[i]>0].mean() for i in range(len(PL)) if (M[i]>0).sum()>0])
    bl_pf  = np.array([be[i][M[i]>0].mean() for i in range(len(BL)) if (M[i]>0).sum()>0])

    alt_errors   = np.abs(PA - TA)[M > 0]
    imp_mean     = (1 - gru_pf.mean()   / bl_pf.mean())   * 100
    imp_median   = (1 - np.median(gru_pf) / np.median(bl_pf)) * 100

    print("\n" + "="*55)
    print("TEST RESULTS (improved v2)")
    print("="*55)
    print(f"  Position - GRU  : mean {gru_pf.mean()/1e3:.2f} km  "
          f"median {np.median(gru_pf)/1e3:.2f} km")
    print(f"  Position - Base : mean {bl_pf.mean()/1e3:.2f} km  "
          f"median {np.median(bl_pf)/1e3:.2f} km")
    print(f"  Improvement      : {imp_mean:.1f}% (mean)  {imp_median:.1f}% (median)")
    print(f"  Altitude RMSE    : {np.sqrt((alt_errors**2).mean()):.0f} ft")
    print(f"  P90 error        : {np.percentile(gru_pf,90)/1e3:.2f} km")
    print("="*55)

    np.savez_compressed(
        OUTPUT_DIR / "test_predictions.npz",
        pred_lat=PL, pred_lon=PO, pred_alt_ft=PA,
        true_lat=TL, true_lon=TO, true_alt_ft=TA,
        baseline_lat=BL, baseline_lon=BO, mask=M,
        gru_errors_m=ge, baseline_errors_m=be)

    summary = {
        "version":                  "v2_improved",
        "changes":                  [
            "exact haversine loss",
            "sequential GRU decoder",
            "last ADS-B dynamics in decoder context",
            f"hidden size {HIDDEN_SIZE}",
        ],
        "best_epoch":               best_ep,
        "best_val_loss":            float(best_val),
        "test_flights":             len(gru_pf),
        "gru_mean_error_km":        float(gru_pf.mean()/1e3),
        "gru_median_error_km":      float(np.median(gru_pf)/1e3),
        "gru_p90_error_km":         float(np.percentile(gru_pf, 90)/1e3),
        "baseline_mean_error_km":   float(bl_pf.mean()/1e3),
        "baseline_median_error_km": float(np.median(bl_pf)/1e3),
        "improvement_mean_pct":     float(imp_mean),
        "improvement_median_pct":   float(imp_median),
        "altitude_rmse_ft":         float(np.sqrt((alt_errors**2).mean())),
        "model_params":             sum(p.numel() for p in model.parameters()
                                        if p.requires_grad),
    }
    json.dump(summary, open(OUTPUT_DIR / "test_summary.json", "w"), indent=2)
    print(f"\nSaved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
