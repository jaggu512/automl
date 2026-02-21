import pandas as pd


def detect_task_type(df, target):
    """
    Infers task type using target cardinality and data type.
    Returns Classification or Regression.
    """

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataframe must be a non-empty pandas DataFrame.")

    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty column name.")

    if target not in df.columns:
        raise ValueError(f"target column '{target}' does not exist in dataframe.")

    target_series = df[target]
    unique_count = target_series.nunique(dropna=False)

    if target_series.dtype == "object" or unique_count < 20:
        return "Classification"

    return "Regression"


def leaderboard_preview(leaderboard_df, limit=5):
    """
    Converts leaderboard dataframe to JSON-serializable records.
    """

    if leaderboard_df is None or leaderboard_df.empty:
        return []

    preview_df = leaderboard_df.head(limit).copy()
    return preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records")
