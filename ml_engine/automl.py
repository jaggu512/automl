import pandas as pd
import inspect

from pycaret.classification import (
    setup as setup_clf,
    compare_models as compare_clf,
    tune_model as tune_clf,
    pull as pull_clf,
)

from pycaret.regression import (
    setup as setup_reg,
    compare_models as compare_reg,
    tune_model as tune_reg,
    pull as pull_reg,
)


VALID_TASK_TYPES = {"classification", "regression"}


def _normalize_task_type(task_type):
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError("task_type must be 'classification' or 'regression'.")

    normalized = task_type.strip().lower()
    if normalized not in VALID_TASK_TYPES:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    return normalized


def _supported_kwargs(func, kwargs):
    signature = inspect.signature(func)
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def run_automl(df, target, task_type):
    """
    Runs full AutoML pipeline:
    - Preprocessing
    - Model comparison
    - Hyperparameter tuning
    - Returns tuned model + leaderboard
    """

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataframe must be a non-empty pandas DataFrame.")

    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty column name.")

    if target not in df.columns:
        raise ValueError(f"target column '{target}' does not exist in dataframe.")

    task_type = _normalize_task_type(task_type)

    if task_type == "classification":
        class_counts = df[target].value_counts(dropna=False)
        if class_counts.empty or int(class_counts.min()) < 2:
            raise ValueError(
                "Classification target has a class with fewer than 2 rows. "
                "Please remove rare classes or choose Regression."
            )

        common_setup_kwargs = {
            "data": df,
            "target": target,
            "session_id": 123,
            "normalize": True,
            "remove_multicollinearity": True,
            "feature_selection": True,
            "fold": 5,
            "verbose": False,
            "html": False,
            "silent": True,
        }

        setup_clf(
            **_supported_kwargs(
                setup_clf,
                common_setup_kwargs,
            )
        )

        best_model = compare_clf(sort="Accuracy")

        leaderboard = pull_clf().copy()

        tuned_model = tune_clf(
            best_model,
            optimize="Accuracy",
            search_library="scikit-learn",
            choose_better=True,
            fold=5
        )

    else:

        common_setup_kwargs = {
            "data": df,
            "target": target,
            "session_id": 123,
            "normalize": True,
            "remove_multicollinearity": True,
            "feature_selection": True,
            "fold": 5,
            "verbose": False,
            "html": False,
            "silent": True,
        }

        setup_reg(**_supported_kwargs(setup_reg, common_setup_kwargs))

        best_model = compare_reg(sort="R2")

        leaderboard = pull_reg().copy()

        tuned_model = tune_reg(
            best_model,
            optimize="R2",
            search_library="scikit-learn",
            choose_better=True,
            fold=5
        )

    return tuned_model, leaderboard
