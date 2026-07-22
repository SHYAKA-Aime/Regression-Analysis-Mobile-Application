"""FastAPI service for tech-talent salary prediction.

Endpoints
---------
GET  /            -> redirects to the Swagger UI (/docs)
GET  /health      -> liveness + which model is currently served
POST /predict     -> predict salary_in_usd from a validated profile
POST /retrain     -> append optional new data and retrain the model (hot-swap)

Run locally:  uv run uvicorn prediction:app --reload --port 8000
Swagger UI:   http://localhost:8000/docs
"""
from __future__ import annotations

import io
import json
import os
from enum import Enum

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import ml  # shared training / feature-engineering logic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
DATA_PATH = os.path.join(MODEL_DIR, "training_data.csv")

# ---------------------------------------------------------------------------
# Load the trained pipeline + metadata at startup
# ---------------------------------------------------------------------------
STATE: dict = {"model": None, "meta": {}}


def load_model() -> None:
    STATE["model"] = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        STATE["meta"] = json.load(f)


load_model()

# ---------------------------------------------------------------------------
# Enumerations mirror the categories the model was trained on
# ---------------------------------------------------------------------------
class ExperienceLevel(str, Enum):
    EN = "EN"  # Entry / Junior
    MI = "MI"  # Mid
    SE = "SE"  # Senior
    EX = "EX"  # Executive / Principal


class EmploymentType(str, Enum):
    FT = "FT"  # Full-time
    PT = "PT"  # Part-time
    CT = "CT"  # Contract
    FL = "FL"  # Freelance


class CompanySize(str, Enum):
    S = "S"
    M = "M"
    L = "L"


class JobCategory(str, Enum):
    data_engineer = "Data Engineer"
    data_scientist = "Data Scientist"
    data_analyst = "Data Analyst"
    ml_ai_engineer = "ML/AI Engineer"
    management = "Management"
    other = "Other"


class CompanyLocation(str, Enum):
    US = "US"
    GB = "GB"
    CA = "CA"
    ES = "ES"
    IN = "IN"
    DE = "DE"
    Other = "Other"


class RemoteRatio(int, Enum):
    onsite = 0
    hybrid = 50
    remote = 100


# ---------------------------------------------------------------------------
# Request / response schemas with enforced data types AND range constraints
# ---------------------------------------------------------------------------
class SalaryRequest(BaseModel):
    work_year: int = Field(..., ge=2020, le=2027,
                           description="Calendar year of the role (2020–2027).", examples=[2023])
    experience_level: ExperienceLevel = Field(..., description="EN, MI, SE or EX.")
    employment_type: EmploymentType = Field(..., description="FT, PT, CT or FL.")
    job_category: JobCategory = Field(..., description="Role family.")
    company_size: CompanySize = Field(..., description="S, M or L.")
    company_location_grp: CompanyLocation = Field(..., description="US, GB, CA, ES, IN, DE or Other.")
    remote_ratio: RemoteRatio = Field(..., description="0 (on-site), 50 (hybrid) or 100 (remote).")

    model_config = {
        "json_schema_extra": {
            "example": {
                "work_year": 2023,
                "experience_level": "SE",
                "employment_type": "FT",
                "job_category": "Data Scientist",
                "company_size": "M",
                "company_location_grp": "US",
                "remote_ratio": 100,
            }
        }
    }


class SalaryResponse(BaseModel):
    predicted_salary_usd: float = Field(..., description="Predicted annual salary in USD.")
    model_used: str
    currency: str = "USD"


class RetrainResponse(BaseModel):
    status: str
    best_model: str
    n_training_rows: int
    metrics: dict


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tech Talent Salary Prediction API",
    description="Predicts a tech professional's expected annual salary (USD) from their profile. "
                "Supports a talent-management mission: quantifying the market value of skills/experience.",
    version="1.0.0",
)

# CORS is scoped deliberately (not a blanket wildcard). See README for the full rationale:
#  * Origins  -> only the deployed API host, the Flutter web dev server and localhost tooling.
#               (Native mobile apps are NOT browsers and are unaffected by CORS, so they still work.)
#  * Methods  -> only GET, POST, OPTIONS — the verbs this API actually exposes.
#  * Headers  -> only Content-Type (JSON/form bodies); nothing else is needed.
#  * Credentials -> disabled, because the API is stateless (no cookies / sessions).
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",   # Flutter web dev server
    "http://127.0.0.1:8000",
    "https://tech-salary-api.onrender.com",  # deployed API host (update to your Render URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
    max_age=600,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "model_used": STATE["meta"].get("best_model"),
    }


@app.post("/predict", response_model=SalaryResponse)
def predict(payload: SalaryRequest) -> SalaryResponse:
    """Predict the expected annual salary (USD) for one profile."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    row = pd.DataFrame([{
        "work_year": payload.work_year,
        "remote_ratio": int(payload.remote_ratio.value),
        "experience_level": payload.experience_level.value,
        "company_size": payload.company_size.value,
        "employment_type": payload.employment_type.value,
        "job_category": payload.job_category.value,
        "company_location_grp": payload.company_location_grp.value,
    }])[ml.FEATURES]
    try:
        pred = float(STATE["model"].predict(row)[0])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return SalaryResponse(predicted_salary_usd=round(pred, 2),
                          model_used=STATE["meta"].get("best_model", "unknown"))


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile | None = File(default=None)) -> RetrainResponse:
    """Retrain the model.

    Optionally upload a CSV of **new** labelled rows (same columns as the training data,
    including `salary_in_usd`). New rows are appended to the stored dataset and all 4
    models are retrained; the lowest-loss model is hot-swapped into service.
    Call with no file to simply retrain on the existing accumulated data.
    """
    base = pd.read_csv(DATA_PATH)

    if file is not None:
        content = await file.read()
        try:
            new_df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse uploaded CSV: {exc}")
        new_df = ml.engineer_features(new_df)
        missing = [c for c in ml.FEATURES + [ml.TARGET] if c not in new_df.columns]
        if missing:
            raise HTTPException(status_code=422,
                                detail=f"Uploaded data missing required columns: {missing}")
        combined = pd.concat([base, new_df[ml.FEATURES + [ml.TARGET]]], ignore_index=True)
    else:
        combined = base

    if len(combined) < 50:
        raise HTTPException(status_code=422, detail="Not enough data to train (need >= 50 rows).")

    try:
        best_model, best_name, metrics = ml.train_and_select(combined)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}")

    # persist accumulated data + new model, then hot-swap
    combined.to_csv(DATA_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)
    meta = STATE["meta"]
    meta["best_model"] = best_name
    meta["metrics"] = {k: {"Test RMSE": round(v["test_rmse"], 2),
                           "Test MAE": round(v["test_mae"], 2),
                           "Test R2": round(v["test_r2"], 3)} for k, v in metrics.items()}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    load_model()

    return RetrainResponse(status="retrained", best_model=best_name,
                           n_training_rows=len(combined), metrics=metrics)
