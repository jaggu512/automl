import os
import joblib


def get_top3(leaderboard_df):
    """
    Returns top 3 models from leaderboard dataframe.
    """

    if leaderboard_df is None or leaderboard_df.empty:
        raise ValueError("Leaderboard is empty.")

    return leaderboard_df.head(3).copy()


def save_model_pipeline(model, task_type, user_id="default"):
    """
    Saves trained model in structured folder format:
    models/user_<id>/best_model.pkl
    """

    base_path = "models"
    user_folder = os.path.join(base_path, f"user_{user_id}")

    os.makedirs(user_folder, exist_ok=True)

    model_path = os.path.join(user_folder, "best_model.pkl")

    joblib.dump(model, model_path)

    return model_path
