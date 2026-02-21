from datetime import datetime, timezone
import io
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ml_engine.automl import run_automl
from ml_engine.evaluation_utils import summarize_leaderboard
from ml_engine.kaggle_engine import download_dataset, recommend_datasets
from ml_engine.model_selection import get_top3, save_model_pipeline
from ml_engine.prediction_engine import predict
from ml_engine.utils import detect_task_type, leaderboard_preview


app = FastAPI(title="Integrated AutoML API", version="1.0.0")


class KaggleDownloadRequest(BaseModel):
    dataset_ref: str


class PredictRequest(BaseModel):
    user_id: str = "default"
    input_data: dict[str, Any] = Field(default_factory=dict)


class EvaluationSummaryRequest(BaseModel):
    task_type: str
    leaderboard_rows: list[dict[str, Any]]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/kaggle/recommendations")
def kaggle_recommendations(task_type: str, limit: int = 5):
    try:
        return {
            "task_type": task_type,
            "results": recommend_datasets(task_type=task_type, limit=limit),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Kaggle recommendation failed: {exc}") from exc


@app.post("/kaggle/download")
def kaggle_download(request: KaggleDownloadRequest):
    try:
        path = download_dataset(request.dataset_ref)
        return {"dataset_ref": request.dataset_ref, "download_path": path}
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

        return {
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
def predict_model(request: PredictRequest):
    try:
        prediction = predict(user_id=request.user_id, input_data=request.input_data)
        return {"user_id": request.user_id, "prediction": prediction}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/evaluate/summary")
def evaluate_summary(request: EvaluationSummaryRequest):
    try:
        leaderboard_df = pd.DataFrame(request.leaderboard_rows)
        summary = summarize_leaderboard(leaderboard_df, request.task_type)
        return {"task_type": request.task_type, "summary_metrics": summary}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation summary failed: {exc}") from exc
