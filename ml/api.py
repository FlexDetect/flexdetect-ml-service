# api.py (replace your current file with this)
import json
import math
import os
import sys
import traceback
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from pandas import Timestamp
import requests
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model import detect_dr

# -------- Logging --------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("flexdetect-ml")

# -------- Configuration (env-driven) --------
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL")  # e.g. "https://my-dataservice.example.com/api"
if not DATA_SERVICE_URL:
    # don't crash at import in dev; fail fast at runtime instead
    DATA_SERVICE_URL = None

DATA_SERVICE_TIMEOUT = float(os.getenv("DATA_SERVICE_TIMEOUT_SEC", "10"))
MAX_ROWS = int(os.getenv("ML_MAX_ROWS", "200_000"))  # defensive upper bound

# -------- App setup --------
app = FastAPI(title="FlexDetect ML Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://flexdetect-frontend.azurewebsites.net",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.on_event("startup")
def startup_event():
    logger.info("ML service starting. DATA_SERVICE_URL=%s", DATA_SERVICE_URL)


@app.get("/_health")
def health():
    return {"ok": True, "DATA_SERVICE_URL": DATA_SERVICE_URL}


# -------- Request models --------
class DetectRequestIds(BaseModel):
    dataset_id: int
    power_measurement_id: int
    feature_measurement_ids: List[int]


# -------- Helpers --------
def _extract_value_from_row(r: Dict[str, Any]) -> Optional[float]:
    """Return numeric representation from row values or None."""
    if r.get("valueFloat") is not None:
        return float(r["valueFloat"])
    if r.get("valueInt") is not None:
        return float(r["valueInt"])
    if r.get("valueBool") is not None:
        # stored as 0/1 in DB according to your schema
        return float(r["valueBool"])
    return None


def _pivot_rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert dataservice measurement rows into a timestamp-indexed dataframe.
    Column names are measurementName.id as strings.
    """
    records = []
    for r in rows:
        mn = r.get("measurementNameIdMeasurementName") or {}
        mid = mn.get("id")
        if mid is None:
            # skip malformed rows
            continue
        col = str(mid)
        val = _extract_value_from_row(r)
        records.append({"timestamp": r["timestamp"], col: val})

    if not records:
        return pd.DataFrame(columns=["timestamp"])

    df = pd.DataFrame(records)
    # keep first value per timestamp/column (pivot)
    df = df.groupby("timestamp").first().reset_index()
    # parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # drop invalid timestamps
    df = df.dropna(subset=["timestamp"])
    # convert boolean-like columns (pandas may already have floats)
    for c in df.columns:
        if c == "timestamp":
            continue
        # attempt numeric cast
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def _safe_float(v, default=None):
    """Return a plain Python float or None for NaN/inf/unconvertible values."""
    try:
        f = float(v)
    except Exception:
        return default
    if math.isfinite(f):
        return f
    return default

def sanitize_timeseries(ts_records):
    """Sanitize list-of-dicts timeseries produced from pandas so JSON is safe."""
    out = []
    for r in ts_records:
        rec = {}
        ts = r.get("timestamp")
        if isinstance(ts, (pd.Timestamp, Timestamp)):
            rec["timestamp"] = ts.isoformat()
        else:
            rec["timestamp"] = str(ts) if ts is not None else None

        # numeric columns: power, baseline_power, dr_flag, dr_capacity_kw
        rec["power"] = _safe_float(r.get("power"), default=None)
        rec["baseline_power"] = _safe_float(r.get("baseline_power"), default=None)

        # dr_flag convert defensively to int (or 0)
        try:
            rec["dr_flag"] = int(r.get("dr_flag")) if r.get("dr_flag") is not None else 0
        except Exception:
            rec["dr_flag"] = 0

        rec["dr_capacity_kw"] = _safe_float(r.get("dr_capacity_kw"), default=0.0)

        out.append(rec)
    return out

def sanitize_events(events):
    out = []
    for e in events:
        out.append({
            "start": e["start"].isoformat() if e.get("start") is not None else None,
            "end": e["end"].isoformat() if e.get("end") is not None else None,
            "flag": int(e["flag"]),
            "energy_kwh": (
                float(e["energy_kwh"])
                if e.get("energy_kwh") is not None
                and not math.isnan(e["energy_kwh"])
                and not math.isinf(e["energy_kwh"])
                else 0.0
            ),
        })
    return out

# -------- Endpoint --------
@app.post("/detect")
def detect(req: DetectRequestIds, request: Request, authorization: Optional[str] = Header(None)):
    if DATA_SERVICE_URL is None:
        raise HTTPException(status_code=500, detail="DATA_SERVICE_URL not configured in environment")
    # safe logging of headers
    try:
        logger.info("Incoming detect request headers: %s", dict(request.headers))
    except Exception:
        logger.exception("Failed logging request headers")

    # fetch rows from data service (server-to-server)
    try:
        headers = {}
        if authorization:
            # forward user token to dataservice if present (use CAUTION in prod)
            headers["Authorization"] = authorization

        url = f"{DATA_SERVICE_URL}/datasets/{req.dataset_id}/measurements"
        logger.info("Requesting data service url=%s", url)
        resp = requests.get(url, headers=headers, timeout=DATA_SERVICE_TIMEOUT)
    except requests.RequestException as e:
        logger.exception("Failed to reach data service: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to reach data service: {str(e)}")

    if resp.status_code != 200:
        # log body for debugging
        body = None
        try:
            body = resp.text
        except Exception:
            body = "<unreadable body>"
        logger.error("Data service returned %s: %s", resp.status_code, body)
        raise HTTPException(status_code=502, detail=f"Data service returned {resp.status_code}")

    try:
        rows = resp.json()
    except Exception:
        logger.exception("Failed to parse JSON from data service response")
        raise HTTPException(status_code=502, detail="Failed to parse data service response")

    # helpful debug: log schema sample
    try:
        logger.info("Data service returned %d rows; sample: %s", len(rows), rows[:3] if isinstance(rows, list) else str(type(rows)))
    except Exception:
        logger.exception("Failed to log sample rows")

    if not rows:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    if len(rows) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"Dataset too large ({len(rows)} rows).")

    # pivot rows -> dataframe
    try:
        df = _pivot_rows_to_dataframe(rows)
    except Exception as e:
        logger.exception("Failed to pivot dataset")
        raise HTTPException(status_code=500, detail=f"Failed to pivot dataset: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid timestamped measurements found")

    # ensure power column exists
    power_col = str(req.power_measurement_id)
    if power_col not in df.columns:
        raise HTTPException(status_code=400, detail="Power signal missing in dataset")

    # set the power column for the model
    df["power"] = df[power_col]
    df = df.dropna(subset=["power"]).reset_index(drop=True)

    # collect selected feature columns (as strings)
    selected_features = []
    for fid in req.feature_measurement_ids:
        col = str(fid)
        if col in df.columns:
            selected_features.append(col)

    # run ML with defensive exception capture
    try:
        model_df = df[["timestamp", "power"] + selected_features].copy()
        df_out, events, model_features = detect_dr(model_df)
    except Exception as e:
        logger.exception("ML processing failed")
        raise HTTPException(status_code=500, detail=f"ML processing failed: {str(e)}")
    # Build timeseries output for the frontend (stringify timestamps)
    timeseries_cols = ["timestamp", "power", "baseline_power", "dr_flag", "dr_capacity_kw"]
    timeseries_out = []
    if all(c in df_out.columns for c in timeseries_cols):
        ts = df_out[timeseries_cols].copy()
        # keep original timestamp dtype; sanitize later
        ts_records = ts.to_dict(orient="records")
        # sanitize to remove NaN/Inf and convert Timestamp -> ISO strings
        timeseries_out = sanitize_timeseries(ts_records)

    # sanitize events & timeseries (events_out already exists)
    events_out = sanitize_events(events)

    # Compute total DR energy (kWh) safely using energy_kwh in events
    total_dr_energy_kwh = 0.0
    for ev in events_out:
        # _safe_float already defined earlier in file — returns float or None
        val = _safe_float(ev.get("energy_kwh"), default=0.0)
        total_dr_energy_kwh += val

    # Build metadata/summary that frontend can rely on
    metadata = {
        "selected_features": model_features,
        "num_events": len(events_out),
    }

    summary = {
        "event_count": len(events_out),
        "total_dr_energy_kwh": float(total_dr_energy_kwh),
    }

    logger.info("Returning %d events and %d timeseries points (total_energy_kwh=%.3f)",
                len(events_out), len(timeseries_out), total_dr_energy_kwh)

    return {
        "metadata": metadata,
        "summary": summary,
        "events": events_out,
        "timeseries": timeseries_out,
    }

