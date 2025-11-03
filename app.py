from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import json

app = FastAPI(title="Edge Impulse Custom DSP Block (Motion Stats + Spectral)")

# ---------- Request model ----------
class DSPRequestBody(BaseModel):
    features: List[float]
    axes: List[str]
    sampling_freq: float
    draw_graphs: bool
    project_id: int
    implementation_version: int
    params: Dict[str, Any] = {}
    calculate_performance: Optional[bool] = False
    named_axes: Optional[Dict[str, Any]] = None

# ---------- Feature helpers ----------
def _basic_stats(x: np.ndarray):
    """Mean, variance, std, min, max, RMS, zero-crossing rate."""
    if x.size == 0:
        return dict(mean=0.0, var=0.0, std=0.0, min=0.0, max=0.0, rms=0.0, zcr=0.0)
    mean = float(np.mean(x))
    var  = float(np.var(x))
    std  = float(np.std(x))
    mn   = float(np.min(x))
    mx   = float(np.max(x))
    rms  = float(np.sqrt(np.mean(x**2)))
    if x.size > 1:
        s = np.sign(x); s[s == 0] = 1
        zcr = float(np.mean(s[1:] * s[:-1] < 0))
    else:
        zcr = 0.0
    return dict(mean=mean, var=var, std=std, min=mn, max=mx, rms=rms, zcr=zcr)

def _spectral_feats(x: np.ndarray, fs: float, low_hz: float, high_hz: float):
    """Band energy fraction, dominant frequency, spectral centroid (Hz)."""
    out = dict(band_energy=0.0, dominant_freq_hz=0.0, spectral_centroid_hz=0.0)
    if x.size == 0 or fs <= 0:
        return out
    # Window to reduce leakage
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    P = np.abs(X)**2
    f = np.fft.rfftfreq(x.size, d=1.0/fs)

    total = float(np.sum(P)) + 1e-12
    mask = (f >= low_hz) & (f <= high_hz)
    out["band_energy"] = float(np.sum(P[mask]) / total)

    kmax = int(np.argmax(P))
    out["dominant_freq_hz"] = float(f[kmax])
    out["spectral_centroid_hz"] = float(np.sum(f * P) / total)
    return out

def compute_motion_features(flat: np.ndarray, axes_names, fs: float, params: dict):
    """
    Build features for motion data from a multi-axis window.
    - axis_mode: 'magnitude' (default) or 'per_axis'
    - band_low_hz / band_high_hz: spectral band for band_energy
    Note: EI converts dashed param keys (band-low-hz) to underscores (band_low_hz) in params.
    """
    low_hz  = float(params.get("band_low_hz", 0.5))
    high_hz = float(params.get("band_high_hz", 5.0))
    axis_mode = str(params.get("axis_mode", "magnitude")).lower()

    n_axes = len(axes_names) if axes_names else 1
    flat = np.asarray(flat, dtype=np.float32)

    # Split equally per axis if multi-axis
    if n_axes > 1:
        L = flat.size // n_axes if n_axes > 0 else flat.size
        parts = [flat[i*L:(i+1)*L] for i in range(n_axes)]
    else:
        parts = [flat]

    labels: List[str] = []
    values: List[float] = []

    def add(prefix: str, d: Dict[str, float]):
        for k, v in d.items():
            labels.append(f"{prefix}_{k}")
            values.append(float(v))

    if axis_mode == "magnitude" and len(parts) >= 2:
        # magnitude = sqrt(x^2 + y^2 + z^2 + ...)
        mag = np.sqrt(np.sum([p**2 for p in parts], axis=0))
        add("mag", _basic_stats(mag))
        add("mag", _spectral_feats(mag, fs, low_hz, high_hz))
    else:
        ax_names = axes_names if axes_names else [f"ax{i}" for i in range(len(parts))]
        for name, sig in zip(ax_names, parts):
            add(name, _basic_stats(sig))
            add(name, _spectral_feats(sig, fs, low_hz, high_hz))

    return labels, values

# ---------- Endpoints ----------
@app.get("/")
async def root():
    return PlainTextResponse("Custom Motion Stats + Spectral (FastAPI)")

@app.get("/parameters")
async def get_parameters():
    with open("parameters.json", "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))

@app.post("/run")
async def run(body: DSPRequestBody):
    x = np.asarray(body.features, dtype=np.float32)
    fs = float(body.sampling_freq)
    params = body.params or {}  # expects: band_low_hz, band_high_hz, axis_mode

    labels, values = compute_motion_features(
        flat=x,
        axes_names=body.axes or [],
        fs=fs,
        params=params
    )

    graphs = []
    # Optional graph: magnitude spectrum if multi-axis, otherwise single-axis spectrum
    if body.draw_graphs and x.size > 0 and fs > 0:
        if (body.axes and len(body.axes) >= 2):
            n_axes = len(body.axes)
            L = x.size // n_axes if n_axes > 0 else x.size
            parts = [x[i*L:(i+1)*L] for i in range(n_axes)]
            sig = np.sqrt(np.sum([p**2 for p in parts], axis=0))
            gname = "Magnitude spectrum"
        else:
            sig = x
            gname = "Power spectrum"
        f = np.fft.rfftfreq(sig.size, d=1.0/fs).tolist()
        P = (np.abs(np.fft.rfft(sig))**2).tolist()
        graphs.append({"name": gname, "type": "xy", "x": f, "y": P})

    resp = {
        "success": True,
        "features": values,
        "graphs": graphs,
        "labels": labels,
        "benchmark_fw_hash": "",
        "output_config": {"type": "flat", "shape": [len(values)]},
        "state_string": None
    }
    return JSONResponse(content=resp)

@app.post("/batch")
async def batch(bodies: List[DSPRequestBody]):
    results = []
    for body in bodies:
        x = np.asarray(body.features, dtype=np.float32)
        fs = float(body.sampling_freq)
        params = body.params or {}
        labels, values = compute_motion_features(
            flat=x,
            axes_names=body.axes or [],
            fs=fs,
            params=params
        )
        results.append({
            "success": True,
            "features": values,
            "graphs": [],
            "labels": labels,
            "benchmark_fw_hash": "",
            "output_config": {"type": "flat", "shape": [len(values)]},
            "state_string": None
        })
    return JSONResponse(content=results)
