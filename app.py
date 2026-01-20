import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, pull, plot_model

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="Learnset – Intelligent AutoML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================
# GLOBAL STYLING (FROM YOUR FIRST CODE)
# ===================================================
st.markdown("""
<style>
.stApp { background-color: #F8FAFC; }

[data-testid="stSidebar"] {
    background-color: #E0F2FE;
    border-right: 2px solid #E5E7EB;
    padding: 14px;
}

.sidebar-card {
    background: white;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 12px;
    border-left: 5px solid #3B82F6;
}

.active-menu {
    background-color: #2563EB;
    color: white;
    padding: 12px;
    border-radius: 10px;
    font-weight: 600;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ===================================================
# SESSION STATE DEFAULTS
# ===================================================
defaults = {
    "page": "Home",
    "project_title": "",
    "project_description": "",
    "dataset_option": None,
    "df": None,
    "dataset_loaded": False,
    "target": None,
    "task": None,
    "automl_ready": False,
    "best_model": None,
    "leaderboard": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================================================
# SIDEBAR
# ===================================================
with st.sidebar:

    st.markdown("## 🧠 Learnset")
    st.caption("Beginner-friendly AutoML system")
    st.markdown("---")

    # Navigation
    st.markdown("### 📍 Navigation")
    st.session_state.page = st.radio(
        "",
        ["Home", "Data Setup", "AutoML & Results"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 System Status")

    def status_card(title, value):
        st.markdown(
            f"""
            <div class="sidebar-card">
                <b>{title}</b><br>
                {value}
            </div>
            """,
            unsafe_allow_html=True
        )

    status_card("Project", "Defined" if st.session_state.project_title else "Not defined")
    status_card("Dataset", "Loaded" if st.session_state.dataset_loaded else "Not selected")
    status_card("AutoML", "Initialized" if st.session_state.automl_ready else "Not ready")
    status_card("Model", "Trained" if st.session_state.best_model is not None else "Not trained")

# ===================================================
# MAIN CONTENT
# ===================================================

# ======================= HOME =======================
if st.session_state.page == "Home":

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#3B82F6,#60A5FA);
        padding:28px;
        border-radius:18px;
        color:white;
        font-size:26px;
        font-weight:700;">
        Intelligent AutoML System
        <p style="font-size:16px;font-weight:400;">
        Describe your problem. Learnset guides you step-by-step.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.form("project_form"):

        st.subheader("📌 Project Information")

        st.session_state.project_title = st.text_input(
            "Project Title",
            st.session_state.project_title,
            placeholder="Student Performance Prediction"
        )

        st.session_state.project_description = st.text_area(
            "Problem Description",
            st.session_state.project_description,
            placeholder="Predict student performance using attendance and internal marks.",
            height=120
        )

        st.subheader("📂 Dataset Availability")

        st.session_state.dataset_option = st.radio(
            "Do you already have a dataset?",
            ["Yes", "No"]
        )

        submit = st.form_submit_button("Save & Continue")

    if submit:
        if not st.session_state.project_title or not st.session_state.project_description:
            st.error("Please fill all project details.")
        else:
            st.success("Project details saved.")
            if st.session_state.dataset_option == "No":
                st.info(
                    "The system will suggest suitable datasets "
                    "based on your problem description."
                )

# ==================== DATA SETUP ====================
elif st.session_state.page == "Data Setup":

    st.title("📁 Data Setup")

    if not st.session_state.project_title:
        st.warning("Complete the Home page first.")
    else:

        if st.session_state.dataset_option == "Yes":

            file = st.file_uploader("Upload CSV dataset", type=["csv"])

            if file:
                st.session_state.df = pd.read_csv(file)
                st.session_state.dataset_loaded = True

                st.success("Dataset uploaded successfully")
                st.dataframe(st.session_state.df.head(), use_container_width=True)

                st.write(
                    f"Rows: {st.session_state.df.shape[0]} | "
                    f"Columns: {st.session_state.df.shape[1]}"
                )

        else:
            st.subheader("📌 Suggested Dataset")

            st.markdown("""
            **Recommended Dataset:** Student Performance Dataset  
            
            **Why this dataset?**
            - Matches your problem description  
            - Structured and beginner-friendly  
            - Suitable for classification/regression tasks  

            👉 For demo purposes, upload a sample CSV matching this dataset.
            """)

# ================= AUTO ML & RESULTS =================
elif st.session_state.page == "AutoML & Results":

    st.title("🚀 AutoML & Results")

    if not st.session_state.dataset_loaded:
        st.warning("Please upload a dataset first.")
    else:

        st.subheader("🎯 Target Selection")

        st.session_state.target = st.selectbox(
            "Select target column",
            st.session_state.df.columns
        )

        unique_vals = st.session_state.df[
            st.session_state.target
        ].nunique()

        st.session_state.task = (
            "Classification" if unique_vals < 20 else "Regression"
        )

        st.info(f"Detected Task: {st.session_state.task}")

        if not st.session_state.automl_ready:
            if st.button("Initialize AutoML"):
                setup(
                    data=st.session_state.df,
                    target=st.session_state.target,
                    verbose=False,
                    html=False
                )
                st.session_state.automl_ready = True
                st.success("AutoML initialized")

        if st.session_state.automl_ready:

            if st.button("Run AutoML"):
                with st.spinner("Training models..."):
                    st.session_state.best_model = compare_models()
                    st.session_state.leaderboard = pull()
                st.success("Training completed")

            if st.session_state.leaderboard is not None:

                st.subheader("📊 Model Comparison")
                st.dataframe(st.session_state.leaderboard.head(5), use_container_width=True)

                st.subheader("💡 Explainability")
                plot_model(
                    st.session_state.best_model,
                    plot="feature",
                    display_format="streamlit"
                )

                st.subheader("📄 Report")
                report_text = f"""
LEARNSET AUTOML REPORT
---------------------
Project: {st.session_state.project_title}
Task: {st.session_state.task}
Target: {st.session_state.target}

Best Model:
{st.session_state.leaderboard.iloc[0]}
"""
                st.download_button(
                    "Download Report",
                    report_text,
                    "Learnset_Report.txt"
                )
