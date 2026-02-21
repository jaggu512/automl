import os
import joblib
import pandas as pd


def _to_python_scalar(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_model(user_id="default"):
    """
    Loads saved model for given user.
    """

    model_path = os.path.join("models", f"user_{user_id}", "best_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train model first.")

    return joblib.load(model_path)


def _expected_feature_names(model):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return None
    return [str(name) for name in list(feature_names)]


def _aligned_input_frame(model, input_data):
    expected = _expected_feature_names(model)
    if not expected:
        return pd.DataFrame([input_data])

    input_keys = set(input_data.keys())
    expected_keys = set(expected)

    missing = [name for name in expected if name not in input_keys]
    extra = sorted([name for name in input_keys if name not in expected_keys])

    if missing:
        missing_preview = ", ".join(missing[:8])
        raise ValueError(
            "Input features do not match model schema. "
            f"Missing: [{missing_preview}]. "
            "Use the same dataset schema used for training or retrain the model."
        )

    aligned_row = {name: input_data[name] for name in expected}
    return pd.DataFrame([aligned_row])


def predict(user_id="default", input_data=None):
    """
    Accepts input data as dictionary and returns prediction.
    """

    if input_data is None or not isinstance(input_data, dict) or not input_data:
        raise ValueError("Input data is required.")

    model = load_model(user_id)

    input_df = _aligned_input_frame(model, input_data)

    prediction = model.predict(input_df)
    if prediction is None or len(prediction) == 0:
        raise ValueError("Model did not return predictions.")

    return _to_python_scalar(prediction[0])
