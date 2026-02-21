import time

import pandas as pd
import streamlit as st

from ml_engine.automl import run_automl
from ml_engine.evaluation_utils import summarize_leaderboard
from ml_engine.kaggle_engine import download_dataset, recommend_datasets
from ml_engine.model_selection import get_top3, save_model_pipeline
from ml_engine.prediction_engine import predict
from ml_engine.utils import detect_task_type


st.set_page_config(
    page_title="Learnset | Integrated AutoML",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp {
    background-color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #F0F9FF;
    border-right: 1px solid #E2E8F0;
}

.gradient-header {
    background: linear-gradient(90deg, #3B82F6, #2563EB);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
}

.active-menu {
    background-color: #2563EB;
    color: white;
    padding: 12px;
    border-radius: 8px;
    text-align: left;
    font-weight: 600;
    margin-bottom: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)


def parse_input_value(raw_value):
    value = str(raw_value).strip()
    if value == "":
        return value

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def metric_display_value(metric_value):
    if metric_value is None:
        return "N/A"
    return f"{metric_value:.4f}"


def reset_training_outputs():
    st.session_state.best_model = None
    st.session_state.leaderboard = None
    st.session_state.top3_models = None
    st.session_state.summary_metrics = None
    st.session_state.model_path = None


defaults = {
    "page": "Home",
    "project_title": "",
    "project_desc": "",
    "df": None,
    "target": None,
    "task_type": None,
    "data_source": "Upload CSV",
    "setup_done": False,
    "best_model": None,
    "leaderboard": None,
    "top3_models": None,
    "summary_metrics": None,
    "model_path": None,
    "kaggle_results": None,
    "prediction_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


with st.sidebar:
    st.markdown("## 🔬 **Learnset**")
    st.caption("Integrated AutoML System")
    st.markdown("---")

    pages = ["Home", "Data Setup", "AutoML & Results", "Prediction & Evaluation"]
    for page in pages:
        if st.session_state.page == page:
            st.markdown(f'<div class="active-menu">{page}</div>', unsafe_allow_html=True)
        else:
            if st.button(page, key=f"nav_{page}", width="stretch"):
                st.session_state.page = page
                st.rerun()

    st.markdown("---")
    st.markdown("### System Status")
    st.write("**Dataset:**")
    if st.session_state.df is not None:
        st.success("Loaded")
    else:
        st.warning("Not Loaded")

    st.write("**Training:**")
    if st.session_state.best_model is not None:
        st.success("Complete")
    else:
        st.info("Pending")


main_page = st.session_state.page

if main_page == "Home":
    st.markdown('<div class="gradient-header">Welcome to Integrated AutoML</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Define Your Problem")
        st.info("Start with project details, then load data and train automatically.")

        with st.form("project_form"):
            title = st.text_input(
                "Project Title",
                value=st.session_state.project_title,
                placeholder="e.g. Student Marks Prediction",
            )
            desc = st.text_area(
                "Problem Description",
                value=st.session_state.project_desc,
                placeholder="Describe what you want to predict...",
                height=150,
            )

            st.write("---")
            data_source = st.radio(
                "Dataset Availability",
                ["Upload CSV", "Kaggle Recommendation"],
                horizontal=True,
            )

            submitted = st.form_submit_button("Save & Continue", type="primary")
            if submitted:
                if title and desc:
                    st.session_state.project_title = title
                    st.session_state.project_desc = desc
                    st.session_state.data_source = data_source
                    st.success("Project initialized. Move to Data Setup.")
                    time.sleep(1)
                    st.session_state.page = "Data Setup"
                    st.rerun()
                else:
                    st.error("Please fill in all fields.")

    with col2:
        st.markdown("### Workflow")
        st.markdown(
            """
        <div style="background:white; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <b>1. Define Problem</b><br>
            Set title and objective.<br><br>
            <b>2. Connect Data</b><br>
            Upload CSV or use Kaggle recommendations.<br><br>
            <b>3. Train Automatically</b><br>
            PyCaret compares and tunes models.<br><br>
            <b>4. Predict and Evaluate</b><br>
            Use saved model and standardized metrics.
        </div>
        """,
            unsafe_allow_html=True,
        )

elif main_page == "Data Setup":
    st.markdown('<div class="gradient-header">Data Setup</div>', unsafe_allow_html=True)

    if not st.session_state.project_title:
        st.warning("Please define your project in Home first.")
    else:
        st.markdown(f"**Project:** {st.session_state.project_title}")

        if st.session_state.get("data_source", "Upload CSV") == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
            if uploaded_file is not None:
                st.session_state.df = pd.read_csv(uploaded_file)
                reset_training_outputs()
                st.success("Dataset uploaded.")
        else:
            st.subheader("Kaggle Dataset Recommendation")
            task_select = st.selectbox("Select Task Type", ["Classification", "Regression"])
            if st.button("Get Kaggle Datasets"):
                try:
                    st.session_state.kaggle_results = recommend_datasets(task_select)
                except Exception as exc:
                    st.error(f"Unable to fetch Kaggle datasets: {exc}")

            if st.session_state.kaggle_results:
                for ds in st.session_state.kaggle_results:
                    col1, col2 = st.columns([4, 1])
                    col1.write(ds["title"])
                    if col2.button("Download", key=ds["ref"]):
                        try:
                            path = download_dataset(ds["ref"])
                            st.success(f"Downloaded to {path}")
                        except Exception as exc:
                            st.error(f"Download failed: {exc}")

        if st.session_state.df is not None:
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Rows", st.session_state.df.shape[0])
            m2.metric("Total Columns", st.session_state.df.shape[1])
            m3.metric("Missing Values", int(st.session_state.df.isnull().sum().sum()))

            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head(), width="stretch")

            st.markdown("---")
            st.subheader("Target Configuration")
            col_target, col_info = st.columns(2)

            with col_target:
                selected_target = st.selectbox("Select Target Column", st.session_state.df.columns)
                st.session_state.target = selected_target

            with col_info:
                if selected_target:
                    try:
                        task = detect_task_type(st.session_state.df, selected_target)
                        st.session_state.task_type = task
                        st.session_state.setup_done = True
                        st.info(f"Detected Task: **{task}**")
                    except Exception as exc:
                        st.session_state.setup_done = False
                        st.error(f"Target validation failed: {exc}")

                    if st.button("Proceed to Training", type="primary"):
                        st.session_state.page = "AutoML & Results"
                        st.rerun()

elif main_page == "AutoML & Results":
    st.markdown('<div class="gradient-header">AutoML & Results</div>', unsafe_allow_html=True)

    if not st.session_state.setup_done:
        st.warning("Please upload data and select a valid target in Data Setup first.")
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Training Configuration")
            st.write(f"Target: **{st.session_state.target}**")
            st.write(f"Task: **{st.session_state.task_type}**")

            if st.button("Start AutoML", type="primary"):
                with st.spinner("PyCaret is training and tuning models..."):
                    try:
                        best_model, leaderboard = run_automl(
                            st.session_state.df,
                            st.session_state.target,
                            st.session_state.task_type,
                        )
                        model_path = save_model_pipeline(
                            best_model,
                            st.session_state.task_type,
                            user_id="default",
                        )

                        top3_models = get_top3(leaderboard)
                        summary_metrics = summarize_leaderboard(leaderboard, st.session_state.task_type)

                        st.session_state.best_model = best_model
                        st.session_state.leaderboard = leaderboard
                        st.session_state.top3_models = top3_models
                        st.session_state.summary_metrics = summary_metrics
                        st.session_state.model_path = model_path
                        st.success("Training completed and model saved.")
                    except Exception as exc:
                        st.error(f"AutoML failed: {exc}")

        with col2:
            st.info("This will run preprocessing, model comparison, and hyperparameter tuning.")

        if st.session_state.leaderboard is not None:
            st.markdown("---")
            st.subheader("Model Leaderboard (Top 5)")
            st.dataframe(st.session_state.leaderboard.head(5), width="stretch")

            st.markdown("### Top 3 Models")
            st.dataframe(st.session_state.top3_models, width="stretch")

            st.markdown("### Standardized Summary")
            metrics = st.session_state.summary_metrics or {}
            if metrics:
                columns = st.columns(len(metrics))
                for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                    columns[idx].metric(metric_name, metric_display_value(metric_value))
            else:
                st.info("No summary metrics available.")

            best_model_name = str(st.session_state.best_model)
            st.success(f"Best Model: **{best_model_name}**")
            if st.session_state.model_path:
                st.caption(f"Saved model path: `{st.session_state.model_path}`")

            report = f"""
PROJECT REPORT: {st.session_state.project_title}
------------------------------------------------
Problem: {st.session_state.project_desc}
Dataset Rows: {st.session_state.df.shape[0]}
Target: {st.session_state.target}
Task: {st.session_state.task_type}

WINNING MODEL: {best_model_name}
SUMMARY METRICS: {st.session_state.summary_metrics}

LEADERBOARD (Top 5):
{st.session_state.leaderboard.head(5).to_string()}
"""
            st.download_button("Download Report", report, "report.txt")

elif main_page == "Prediction & Evaluation":
    st.markdown('<div class="gradient-header">Prediction & Evaluation</div>', unsafe_allow_html=True)

    if st.session_state.df is None or st.session_state.target is None:
        st.warning("Load a dataset and choose a target in Data Setup first.")
    else:
        st.subheader("Prediction")
        if st.session_state.best_model is None:
            st.warning("Train a model in this session before running prediction.")

        feature_columns = [
            col for col in st.session_state.df.columns if col != st.session_state.target
        ]

        with st.form("prediction_form"):
            user_id = st.text_input("User ID", value="default")
            st.caption("Provide input values for each feature.")

            input_payload = {}
            for feature in feature_columns:
                default_value = ""
                if not st.session_state.df.empty:
                    sample_value = st.session_state.df.iloc[0][feature]
                    if pd.notna(sample_value):
                        default_value = str(sample_value)
                raw_value = st.text_input(feature, value=default_value, key=f"pred_{feature}")
                if raw_value != "":
                    input_payload[feature] = parse_input_value(raw_value)

            submit_prediction = st.form_submit_button("Run Prediction", type="primary")

        if submit_prediction:
            try:
                if st.session_state.best_model is None:
                    raise ValueError("No trained model in current session. Run AutoML first.")
                prediction = predict(user_id=user_id, input_data=input_payload)
                st.session_state.prediction_result = prediction
                st.success(f"Prediction: **{prediction}**")
                st.caption(f"Model path: `models/user_{user_id}/best_model.pkl`")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        st.markdown("---")
        st.subheader("Evaluation Summary")

        if st.session_state.leaderboard is not None and st.session_state.task_type:
            if st.button("Refresh Summary Metrics"):
                try:
                    st.session_state.summary_metrics = summarize_leaderboard(
                        st.session_state.leaderboard,
                        st.session_state.task_type,
                    )
                except Exception as exc:
                    st.error(f"Unable to refresh summary: {exc}")

            if st.session_state.summary_metrics:
                metrics = st.session_state.summary_metrics
                metric_cols = st.columns(len(metrics))
                for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                    metric_cols[idx].metric(metric_name, metric_display_value(metric_value))
            else:
                st.info("Summary metrics are not available yet. Train a model first.")
        else:
            st.info("Run AutoML first to generate leaderboard-based evaluation metrics.")
