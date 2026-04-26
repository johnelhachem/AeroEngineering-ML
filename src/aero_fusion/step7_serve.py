"""Serve trajectory reconstructions through the API layer."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not installed - GRU inference disabled, using baseline only")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[ERROR] FastAPI not installed. Run: pip install fastapi uvicorn")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP2_ROOT   = PROJECT_ROOT / "artifacts" / "step2_clean"
STEP4_ROOT   = PROJECT_ROOT / "artifacts" / "step4_ml_dataset"
STEP5_ROOT   = PROJECT_ROOT / "artifacts" / "step5_gru"

BEFORE_STEPS    = 64
AFTER_STEPS     = 32
N_SEQ_FEATURES  = 6
MAX_ADSC_WP     = 32
HIDDEN_SIZE     = 160
NUM_LAYERS      = 2
RESAMPLE_SEC    = 60
EARTH_R         = 6_371_000.0
N_LAST_DYN      = N_SEQ_FEATURES

LAT_MEAN, LAT_STD   = 53.0,   8.0
LON_MEAN, LON_STD   = -30.0,  25.0
VEL_MEAN, VEL_STD   = 240.0,  30.0
ALT_MEAN, ALT_STD   = 10500.0, 1000.0

if TORCH_AVAILABLE:
    class TrajectoryGRU(nn.Module):
        def __init__(self, D=N_SEQ_FEATURES, H=HIDDEN_SIZE, L=NUM_LAYERS, p=0.2):
            super().__init__()
            self.before_enc = nn.GRU(D, H, L, batch_first=True, bidirectional=True,
                                      dropout=p if L > 1 else 0.0)
            self.after_enc  = nn.GRU(D, H, L, batch_first=True, bidirectional=False,
                                      dropout=p if L > 1 else 0.0)
            C = 2*H + H + 1 + N_LAST_DYN
            self.ctx_to_h0 = nn.Linear(C, H)
            self.decoder_gru = nn.GRU(3, H, 1, batch_first=True)
            self.decoder_head = nn.Sequential(
                nn.Linear(H, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(p),
                nn.Linear(128, 64), nn.GELU(),
                nn.Linear(64, 3),
            )

            self._init()

        def _init(self):
            for n, p in self.named_parameters():
                if "weight_ih" in n:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in n:
                    nn.init.orthogonal_(p)
                elif "bias" in n:
                    nn.init.zeros_(p)
                elif "weight" in n and p.dim() == 2:
                    nn.init.xavier_uniform_(p)

        def _enc(self, enc, seq, mask):
            lengths = mask.sum(1).long().clamp(min=1).cpu()
            packed  = nn.utils.rnn.pack_padded_sequence(
                seq, lengths, batch_first=True, enforce_sorted=False)
            _, h = enc(packed)
            return h

        def forward(self, b):
            hb = self._enc(self.before_enc, b["before_seq"],  b["before_mask"])
            ha = self._enc(self.after_enc,  b["after_seq"],   b["after_mask"])
            ctx = torch.cat([
                hb[-2],
                hb[-1],
                ha[-1],
                b["gap_norm"].unsqueeze(-1),
                b["last_dyn"],
            ], -1)
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
            pred_alt = res[:, :, 2]
            return pred_lat, pred_lon, pred_alt

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return EARTH_R * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a)))) / 1000.0

def gc_point(lat0, lon0, lat1, lon1, tau: float):
    lat0r,lon0r = math.radians(lat0), math.radians(lon0)
    lat1r,lon1r = math.radians(lat1), math.radians(lon1)
    x0=math.cos(lat0r)*math.cos(lon0r); y0=math.cos(lat0r)*math.sin(lon0r); z0=math.sin(lat0r)
    x1=math.cos(lat1r)*math.cos(lon1r); y1=math.cos(lat1r)*math.sin(lon1r); z1=math.sin(lat1r)
    dot   = max(-1.0, min(1.0, x0*x1+y0*y1+z0*z1))
    omega = math.acos(dot)
    if abs(omega) < 1e-10:
        return lat0, lon0
    s = math.sin(omega)
    w0 = math.sin((1-tau)*omega)/s;  w1 = math.sin(tau*omega)/s
    xp,yp,zp = w0*x0+w1*x1, w0*y0+w1*y1, w0*z0+w1*z1
    n = math.sqrt(xp**2+yp**2+zp**2)
    return math.degrees(math.asin(max(-1,min(1,zp/n)))), math.degrees(math.atan2(yp/n,xp/n))

def track_length_km(lats, lons) -> float:
    if len(lats) < 2: return 0.0
    phi1=np.radians(lats[:-1]); phi2=np.radians(lats[1:])
    dphi=np.radians(lats[1:]-lats[:-1]); dlam=np.radians(lons[1:]-lons[:-1])
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return float((EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))).sum()) / 1000.0

def _safe_float(val) -> float:
    """Convert a value to float or return NaN."""
    if val is None:
        return float("nan")
    try:
        if pd.isna(val):
            return float("nan")
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")

def normalize(val, mean, std) -> float:
    v = _safe_float(val)
    if not math.isfinite(v): return 0.0
    return (v - mean) / std

def resample_track(df: pd.DataFrame, n_steps: int, from_end: bool) -> tuple:
    out  = np.zeros((n_steps, N_SEQ_FEATURES), dtype=np.float32)
    mask = np.zeros(n_steps, dtype=np.float32)
    if df.empty: return out, mask

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty: return out, mask

    if len(df) >= 2:
        start = df["timestamp"].min().floor(f"{RESAMPLE_SEC}s")
        end   = df["timestamp"].max().ceil(f"{RESAMPLE_SEC}s")
        if start == end: end = start + pd.Timedelta(seconds=RESAMPLE_SEC)
        grid  = pd.date_range(start=start, end=end, freq=f"{RESAMPLE_SEC}s")
        cols  = [c for c in ["latitude","longitude","velocity_mps","heading_deg","geoaltitude_m"]
                 if c in df.columns]
        df = (df.set_index("timestamp")[cols].sort_index()
                .reindex(grid)
                .interpolate(method="time", limit_direction="forward", limit_area="inside")
                .reset_index().rename(columns={"index":"timestamp"}))

    df = df.tail(n_steps) if from_end else df.head(n_steps)
    df = df.reset_index(drop=True)
    offset = (n_steps - len(df)) if from_end else 0

    for i, row in df.iterrows():
        dest = offset + i if from_end else i
        if dest >= n_steps: break
        lat = _safe_float(row.get("latitude",  np.nan))
        lon = _safe_float(row.get("longitude", np.nan))
        vel = _safe_float(row.get("velocity_mps",  np.nan))
        hdg = _safe_float(row.get("heading_deg",   np.nan))
        alt = _safe_float(row.get("geoaltitude_m", np.nan))
        out[dest,0] = normalize(lat, LAT_MEAN, LAT_STD)
        out[dest,1] = normalize(lon, LON_MEAN, LON_STD)
        out[dest,2] = normalize(vel, VEL_MEAN, VEL_STD)
        out[dest,3] = math.sin(math.radians(hdg)) if math.isfinite(hdg) else 0.0
        out[dest,4] = math.cos(math.radians(hdg)) if math.isfinite(hdg) else 0.0
        out[dest,5] = normalize(alt, ALT_MEAN, ALT_STD)
        mask[dest]  = 1.0
    return out, mask

_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not TORCH_AVAILABLE:
        return None

    model_path = STEP5_ROOT / "best_model.pt"
    if not model_path.exists():
        print(f"[WARN] Model not found at {model_path} - using baseline only")
        return None

    model = TrajectoryGRU()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    _model_cache = model
    print(f"[INFO] Model loaded from {model_path}")
    return model

def load_flight_catalog():
    """Load the catalog of available flights."""
    catalog_path = STEP4_ROOT / "catalog" / "flight_splits.parquet"
    if not catalog_path.exists():
        catalog_path = STEP2_ROOT / "catalog" / "clean_flights_validated.parquet"
    if not catalog_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(catalog_path)

def reconstruct_flight(segment_id: str, use_gru: bool = True) -> dict[str, Any]:
    """Reconstruct the full trajectory for a given segment_id."""
    t_start = time.time()
    flight_dir = STEP2_ROOT / "flights" / segment_id

    if not flight_dir.exists():
        raise ValueError(f"Flight not found: {segment_id}")

    before_path = flight_dir / "adsb_before_clean.parquet"
    after_path  = flight_dir / "adsb_after_clean.parquet"
    adsc_path   = flight_dir / "adsc_clean.parquet"

    for p in [before_path, after_path]:
        if not p.exists():
            raise ValueError(f"Missing file: {p.name} for {segment_id}")

    before_df = pd.read_parquet(before_path)
    after_df  = pd.read_parquet(after_path)
    adsc_df   = pd.read_parquet(adsc_path) if adsc_path.exists() else pd.DataFrame()

    for df in [before_df, after_df, adsc_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    before_df = before_df.dropna(subset=["timestamp","latitude","longitude"]).sort_values("timestamp")
    after_df  = after_df.dropna(subset=["timestamp","latitude","longitude"]).sort_values("timestamp")

    if before_df.empty or after_df.empty:
        raise ValueError(f"Empty before/after track for {segment_id}")

    b_anc = before_df.iloc[-1]
    a_anc = after_df.iloc[0]
    t0    = pd.Timestamp(b_anc["timestamp"])
    t1    = pd.Timestamp(a_anc["timestamp"])
    dur   = float((t1 - t0).total_seconds())

    if dur <= 0:
        raise ValueError(f"Invalid gap timestamps for {segment_id}")

    anchor_dist_km = haversine_km(
        float(b_anc["latitude"]), float(b_anc["longitude"]),
        float(a_anc["latitude"]), float(a_anc["longitude"]),
    )

    n_gc_steps = max(2, int(dur / RESAMPLE_SEC) + 1)
    baseline_wp = []
    for tau in np.linspace(0, 1, n_gc_steps):
        lat, lon = gc_point(float(b_anc["latitude"]), float(b_anc["longitude"]),
                            float(a_anc["latitude"]), float(a_anc["longitude"]), float(tau))
        ts = t0 + pd.Timedelta(seconds=float(tau * dur))
        baseline_wp.append({"timestamp": ts.isoformat(), "latitude": round(lat,6),
                             "longitude": round(lon,6), "source": "baseline"})

    method = "baseline"
    gru_wp = []

    model = load_model() if use_gru else None

    if model is not None and TORCH_AVAILABLE:
        try:
            before_seq,  before_mask  = resample_track(before_df, BEFORE_STEPS, from_end=True)
            after_seq,   after_mask   = resample_track(after_df,  AFTER_STEPS,  from_end=False)

            if not adsc_df.empty:
                adsc_sorted = adsc_df.sort_values("timestamp").dropna(
                    subset=["timestamp","latitude","longitude"])
                n_wp = min(len(adsc_sorted), MAX_ADSC_WP)
                adsc_tau_arr = np.zeros(MAX_ADSC_WP, dtype=np.float32)
                for j in range(n_wp):
                    elapsed = float((pd.Timestamp(adsc_sorted["timestamp"].iloc[j]) - t0
                                     ).total_seconds())
                    adsc_tau_arr[j] = float(np.clip(elapsed / dur, 0, 1))
                adsc_tau_arr[n_wp:] = 0.0
                adsc_mask_arr = np.zeros(MAX_ADSC_WP, dtype=np.float32)
                adsc_mask_arr[:n_wp] = 1.0
            else:
                n_wp = 8
                adsc_tau_arr  = np.linspace(0, 1, MAX_ADSC_WP, dtype=np.float32)
                adsc_mask_arr = np.zeros(MAX_ADSC_WP, dtype=np.float32)
                adsc_mask_arr[:n_wp] = 1.0

            bl_lat = np.zeros(MAX_ADSC_WP, dtype=np.float32)
            bl_lon = np.zeros(MAX_ADSC_WP, dtype=np.float32)
            for j in range(MAX_ADSC_WP):
                if adsc_mask_arr[j] > 0:
                    lat, lon = gc_point(float(b_anc["latitude"]), float(b_anc["longitude"]),
                                        float(a_anc["latitude"]), float(a_anc["longitude"]),
                                        float(adsc_tau_arr[j]))
                    bl_lat[j], bl_lon[j] = lat, lon

            batch = {
                "before_seq":   torch.tensor(before_seq[None]),
                "before_mask":  torch.tensor(before_mask[None]),
                "after_seq":    torch.tensor(after_seq[None]),
                "after_mask":   torch.tensor(after_mask[None]),
                "adsc_tau":     torch.tensor(adsc_tau_arr[None]),
                "baseline_lat": torch.tensor(bl_lat[None]),
                "baseline_lon": torch.tensor(bl_lon[None]),
                "gap_norm":     torch.tensor([dur / 6000.0], dtype=torch.float32),
                "last_dyn":     torch.tensor(before_seq[None, np.where(before_mask > 0)[0][-1], :]),
            }

            with torch.no_grad():
                pred_lat, pred_lon, _ = model(batch)

            pred_lat_np = pred_lat[0].numpy()
            pred_lon_np = pred_lon[0].numpy()

            for j in range(MAX_ADSC_WP):
                if adsc_mask_arr[j] > 0:
                    ts = t0 + pd.Timedelta(seconds=float(adsc_tau_arr[j] * dur))
                    gru_wp.append({
                        "timestamp": ts.isoformat(),
                        "latitude":  round(float(pred_lat_np[j]), 6),
                        "longitude": round(float(pred_lon_np[j]), 6),
                        "source":    "gru",
                    })

            method = "gru"

        except Exception as exc:
            print(f"[WARN] GRU inference failed for {segment_id}: {exc} - falling back to baseline")
            gru_wp = []
            method = "baseline"

    def df_to_records(df, source):
        return [{"timestamp": pd.Timestamp(r["timestamp"]).isoformat(),
                 "latitude":  round(float(r["latitude"]),6),
                 "longitude": round(float(r["longitude"]),6),
                 "source":    source}
                for _, r in df.iterrows()]

    def densify_waypoints(waypoints: list, t_start_ts, t_end_ts,
                          source_label: str, step_sec: int = 60) -> list:
        """Interpolate between sparse waypoints along great-circle arcs."""
        if not waypoints:
            return []
        all_wp = (
            [{"timestamp": t_start_ts.isoformat(),
              "latitude":  round(float(b_anc["latitude"]),6),
              "longitude": round(float(b_anc["longitude"]),6)}] +
            waypoints +
            [{"timestamp": t_end_ts.isoformat(),
              "latitude":  round(float(a_anc["latitude"]),6),
              "longitude": round(float(a_anc["longitude"]),6)}]
        )
        dense = []
        for i in range(len(all_wp) - 1):
            p0, p1 = all_wp[i], all_wp[i+1]
            ts0 = pd.Timestamp(p0["timestamp"])
            ts1 = pd.Timestamp(p1["timestamp"])
            seg_dur = float((ts1 - ts0).total_seconds())
            if seg_dur <= 0:
                continue
            n_steps = max(2, int(seg_dur / step_sec) + 1)
            for tau in np.linspace(0, 1, n_steps):
                if i > 0 and tau == 0.0:
                    continue  # avoid duplicate at segment boundary
                lat, lon = gc_point(p0["latitude"], p0["longitude"],
                                    p1["latitude"], p1["longitude"], float(tau))
                ts = ts0 + pd.Timedelta(seconds=float(tau * seg_dur))
                dense.append({"timestamp": ts.isoformat(),
                               "latitude":  round(lat, 6),
                               "longitude": round(lon, 6),
                               "source":    source_label})
        return dense

    if method == "gru" and gru_wp:
        dense_gru_fill = densify_waypoints(gru_wp, t0, t1, "gru_dense")
    else:
        dense_gru_fill = densify_waypoints(baseline_wp[:2], t0, t1, "gru_dense") if baseline_wp else []

    dense_baseline_fill = densify_waypoints(baseline_wp, t0, t1, "baseline_dense")

    full_track = (
        df_to_records(before_df, "adsb_before") +
        dense_gru_fill +
        df_to_records(after_df, "adsb_after")
    )

    seen = set()
    unique_track = []
    for pt in full_track:
        if pt["timestamp"] not in seen:
            seen.add(pt["timestamp"])
            unique_track.append(pt)
    unique_track.sort(key=lambda x: x["timestamp"])

    lats = np.array([p["latitude"]  for p in unique_track])
    lons = np.array([p["longitude"] for p in unique_track])
    route_dist_km = track_length_km(lats, lons)

    base_track_pts = (df_to_records(before_df, "adsb_before") +
                      dense_baseline_fill +
                      df_to_records(after_df, "adsb_after"))
    base_lats = np.array([p["latitude"]  for p in base_track_pts])
    base_lons = np.array([p["longitude"] for p in base_track_pts])
    baseline_route_dist_km = track_length_km(base_lats, base_lons)

    elapsed_ms = int((time.time() - t_start) * 1000)

    return {
        "segment_id":              segment_id,
        "method":                  method,
        "gru_available":           model is not None,
        "gap_duration_minutes":    round(dur / 60.0, 2),
        "anchor_distance_km":      round(anchor_dist_km, 2),
        "route_distance_km":       round(route_dist_km, 2),
        "baseline_route_dist_km":  round(baseline_route_dist_km, 2),
        "n_track_points":          len(unique_track),
        "n_gap_waypoints":         len(gru_wp),
        "processing_ms":           elapsed_ms,
        "track":                   unique_track,
        "baseline_track":          base_track_pts,
        "gru_waypoints":           gru_wp,
        "baseline_waypoints":      dense_baseline_fill,
    }

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="AeroFusion Trajectory Reconstruction API",
        description=(
            "Reconstructs complete aircraft trajectories over oceanic gaps "
            "by fusing ADS-B context with a trained GRU model. "
            "Built for the ADS-B + ADS-C fusion project at USJ Lebanon."
        ),
        version="1.0.0",
    )

    @app.on_event("startup")
    async def startup_event():
        load_model()
        print("[INFO] API ready")

    @app.get("/health")
    def health():
        """Health check endpoint."""
        model = load_model()
        return {
            "status": "ok",
            "gru_model_loaded": model is not None,
            "torch_available":  TORCH_AVAILABLE,
            "step2_root":       str(STEP2_ROOT),
        }

    @app.get("/flights")
    def list_flights(split: str | None = None, limit: int = 50):
        """List available flights."""
        catalog = load_flight_catalog()
        if catalog.empty:
            raise HTTPException(status_code=500, detail="Flight catalog not found")

        if split and "split" in catalog.columns:
            catalog = catalog[catalog["split"] == split]

        cols = [c for c in ["segment_id","icao24","split","gap_duration_minutes",
                             "estdepartureairport","estarrivalairport"] if c in catalog.columns]
        result = catalog[cols].head(limit).to_dict(orient="records")
        return {"count": len(result), "flights": result}

    @app.get("/reconstruct/{segment_id}")
    def reconstruct(segment_id: str, use_gru: bool = True):
        """Reconstruct the full trajectory for a given segment_id."""
        try:
            result = reconstruct_flight(segment_id, use_gru=use_gru)
            return JSONResponse(content=result)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reconstruction failed: {e}")

    @app.get("/compare/{segment_id}")
    def compare(segment_id: str):
        """Return GRU and baseline reconstructions for the same flight."""
        try:
            gru_result      = reconstruct_flight(segment_id, use_gru=True)
            baseline_result = reconstruct_flight(segment_id, use_gru=False)
            return JSONResponse(content={
                "segment_id":   segment_id,
                "gru":          gru_result,
                "baseline":     baseline_result,
                "distance_diff_km": round(
                    gru_result["route_distance_km"] - baseline_result["route_distance_km"], 2
                ),
            })
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def main():
    if not FASTAPI_AVAILABLE:
        print("Install FastAPI first: pip install fastapi uvicorn")
        return
    uvicorn.run("aero_fusion.step7_serve:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
