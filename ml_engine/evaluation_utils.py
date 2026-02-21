import pandas as pd


VALID_TASK_TYPES = {"classification", "regression"}


def _normalize_task_type(task_type):
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError("task_type must be 'classification' or 'regression'.")

    normalized = task_type.strip().lower()
    if normalized not in VALID_TASK_TYPES:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    return normalized


def _safe_metric(value):
    if value is None:
        return None

    if pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_leaderboard(leaderboard_df, task_type):
    """
    Returns structured evaluation summary dictionary.
    """

    if leaderboard_df is None or leaderboard_df.empty:
        raise ValueError("Leaderboard is empty.")

    best_row = leaderboard_df.iloc[0]
    task_type = _normalize_task_type(task_type)

    if task_type == "classification":
        return {
            "Accuracy": _safe_metric(best_row.get("Accuracy")),
            "AUC": _safe_metric(best_row.get("AUC")),
            "F1": _safe_metric(best_row.get("F1"))
        }

    else:
        return {
            "MAE": _safe_metric(best_row.get("MAE")),
            "RMSE": _safe_metric(best_row.get("RMSE")),
            "R2": _safe_metric(best_row.get("R2"))
        }
    
