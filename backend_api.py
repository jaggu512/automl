from datetime import datetime, timezone
import io
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ml_engine.automl import run_automl
from ml_engine.db import PredictionLog, SessionLocal, TrainingRun, User, init_db, ping_db
from ml_engine.evaluation_utils import summarize_leaderboard
from ml_engine.kaggle_engine import download_dataset, recommend_datasets
from ml_engine.model_selection import get_top3, save_model_pipeline
from ml_engine.prediction_engine import predict
from ml_engine.security import (
    create_access_token,
    encrypt_payload,
    hash_password,
    verify_password,
    decode_access_token,
)
from ml_engine.utils import detect_task_type, leaderboard_preview


app = FastAPI(title="Integrated AutoML API", version="1.0.0")
DB_STARTUP_ERROR = None


class KaggleDownloadRequest(BaseModel):
    dataset_ref: str


class PredictRequest(BaseModel):
    user_id: str = "default"
    input_data: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummaryRequest(BaseModel):
    task_type: str
    leaderboard_rows: list[dict[str, Any]]


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


def get_db():
    if DB_STARTUP_ERROR is not None:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {DB_STARTUP_ERROR}")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _extract_bearer_token(authorization):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization must be Bearer token.")
    return parts[1]


def get_current_username(
    authorization: str = Header(default=None),
):
    token = _extract_bearer_token(authorization)
    username = decode_access_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return username


@app.on_event("startup")
def on_startup():
    global DB_STARTUP_ERROR
    try:
        init_db()
        DB_STARTUP_ERROR = None
    except Exception as exc:
        DB_STARTUP_ERROR = str(exc)


@app.get("/health")
def health():
    db_ok = False
    db_error = None
    try:
        ping_db()
        db_ok = True
    except Exception as exc:
        db_error = str(exc)

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {"connected": db_ok and DB_STARTUP_ERROR is None, "error": DB_STARTUP_ERROR or db_error},
    }


@app.get("/kaggle/recommendations")
def kaggle_recommendations(
    task_type: str,
    limit: int = 5,
    username: str = Depends(get_current_username),
):
    try:
        return {
            "username": username,
            "task_type": task_type,
            "results": recommend_datasets(task_type=task_type, limit=limit),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kaggle recommendation failed: {exc}") from exc


@app.post("/kaggle/download")
def kaggle_download(
    request: KaggleDownloadRequest,
    username: str = Depends(get_current_username),
):
    try:
        path = download_dataset(request.dataset_ref)
        return {"username": username, "dataset_ref": request.dataset_ref, "download_path": path}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kaggle download failed: {exc}") from exc


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target: str = Form(...),
    task_type: str | None = Form(None),
    user_id: str = Form("default"),
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    try:
        payload = await file.read()
        if not payload:
            raise ValueError("Uploaded CSV file is empty.")

        df = pd.read_csv(io.BytesIO(payload))
        resolved_task = task_type if task_type else detect_task_type(df, target)

        best_model, leaderboard = run_automl(df, target, resolved_task)
        model_path = save_model_pipeline(best_model, resolved_task, user_id=user_id)

        top3 = get_top3(leaderboard)
        summary = summarize_leaderboard(leaderboard, resolved_task)

        top_model_score = None
        if not leaderboard.empty and leaderboard.shape[1] > 1:
            score_value = leaderboard.iloc[0, 1]
            try:
                top_model_score = float(score_value)
            except (TypeError, ValueError):
                top_model_score = None

        db.add(
            TrainingRun(
                username=username,
                user_id=user_id,
                task_type=resolved_task,
                target_column=target,
                dataset_rows=int(df.shape[0]),
                dataset_columns=int(df.shape[1]),
                model_path=model_path,
                top_model_score=top_model_score,
                summary_metrics=summary,
            )
        )
        db.commit()

        return {
            "username": username,
            "user_id": user_id,
            "task_type": resolved_task,
            "model_path": model_path,
            "top3": leaderboard_preview(top3, limit=3),
            "summary_metrics": summary,
            "leaderboard_preview": leaderboard_preview(leaderboard, limit=5),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@app.post("/predict")
def predict_model(
    request: PredictRequest,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    try:
        prediction = predict(user_id=request.user_id, input_data=request.input_data)

        db.add(
            PredictionLog(
                username=username,
                user_id=request.user_id,
                prediction_value=str(prediction),
                encrypted_input=encrypt_payload(request.input_data),
            )
        )
        db.commit()

        return {"username": username, "user_id": request.user_id, "prediction": prediction}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/evaluate/summary")
def evaluate_summary(
    request: EvaluationSummaryRequest,
    username: str = Depends(get_current_username),
):
    try:
        leaderboard_df = pd.DataFrame(request.leaderboard_rows)
        summary = summarize_leaderboard(leaderboard_df, request.task_type)
        return {"username": username, "task_type": request.task_type, "summary_metrics": summary}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation summary failed: {exc}") from exc


@app.post("/auth/register")
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        username = request.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="username is required.")
        if not request.password:
            raise HTTPException(status_code=400, detail="password is required.")

        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise HTTPException(status_code=409, detail="username already exists.")

        user = User(username=username, password_hash=hash_password(request.password))
        db.add(user)
        db.commit()

        return {"status": "registered", "username": username}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registration failed: {exc}") from exc


@app.post("/auth/login")
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        username = request.username.strip()
        user = db.query(User).filter(User.username == username).first()
        if user is None or not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="invalid credentials.")

        access_token = create_access_token(username=user.username)
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}") from exc
