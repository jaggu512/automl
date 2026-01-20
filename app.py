import streamlit as st
import pandas as pd
import time
import os

# --- PYCARET IMPORTS ---
# We import both modules to handle Classification and Regression dynamically
from pycaret.classification import setup as setup_clf, compare_models as compare_clf, pull as pull_clf, save_model as save_clf, load_model as load_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_reg, pull as pull_reg, save_model as save_reg, load_model as load_reg

# ===================================================
# 1. PAGE CONFIGURATION
# ===================================================
st.set_page_config(
    page_title="Learnset | Intelligent AutoML",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================
# 2. GLOBAL STYLING (Professional Look)
# ===================================================
st.markdown("""
<style>
/* Main Background */
.stApp {
    background-color: #F8FAFC;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #F0F9FF;
    border-right: 1px solid #E2E8F0;
}

/* Gradient Header Card */
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

/* Metric Cards */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* Custom Buttons */
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
}

/* Active Menu Item Styling */
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
""", unsafe_allow_html=True)

# ===================================================
# 3. SESSION STATE MANAGEMENT
# ===================================================
defaults = {
    "page": "Home",
    "project_title": "",
    "project_desc": "",
    "df": None,
    "target": None,
    "task_type": None,
    "setup_done": False,
    "best_model": None,
    "leaderboard": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===================================================
# 4. HELPER FUNCTIONS (The "PyCaret Engine")
# ===================================================
def detect_task(df, target):
    """Smartly decides if the problem is Classification or Regression"""
    if df[target].nunique() < 20 or df[target].dtype == 'object':
        return "Classification"
    return "Regression"

def run_pycaret_automl(df, target, task):
    """Runs PyCaret AutoML based on the task type"""
    
    if task == "Classification":
        # 1. Setup (Preprocessing)
        s = setup_clf(data=df, target=target, verbose=False, html=False, session_id=123)
        # 2. Compare Models (Training)
        best_model = compare_clf()
        # 3. Get Results
        leaderboard = pull_clf()
        # 4. Save
        save_clf(best_model, 'best_model_pipeline')
        
    else:
        # 1. Setup (Preprocessing)
        s = setup_reg(data=df, target=target, verbose=False, html=False, session_id=123)
        # 2. Compare Models (Training)
        best_model = compare_reg()
        # 3. Get Results
        leaderboard = pull_reg()
        # 4. Save
        save_reg(best_model, 'best_model_pipeline')
        
    return leaderboard, best_model

# ===================================================
# 5. SIDEBAR NAVIGATION
# ===================================================
with st.sidebar:
    st.markdown("## 🔬 **Learnset**")
    st.caption("Intelligent AutoML System")
    st.markdown("---")
    
    # Custom Menu Buttons
    pages = ["Home", "Data Setup", "AutoML & Results"]
    
    for p in pages:
        if st.session_state.page == p:
            st.markdown(f'<div class="active-menu">{p}</div>', unsafe_allow_html=True)
        else:
            if st.button(p, key=f"nav_{p}", use_container_width=True):
                st.session_state.page = p
                st.rerun()
                
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    # Status Indicators
    st.write("**Dataset:**")
    if st.session_state.df is not None:
        st.success("Loaded")
    else:
        st.warning("Not Loaded")
        
    st.write("**Training:**")
    if st.session_state.best_model:
        st.success("Complete")
    else:
        st.info("Pending")

# ===================================================
# 6. MAIN CONTENT
# ===================================================
main_page = st.session_state.page

# --- PAGE 1: HOME ---
if main_page == "Home":
    st.markdown('<div class="gradient-header">🏠 Welcome to Intelligent AutoML</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Define Your Problem")
        st.info("Start by telling us about your project. The AI will guide you based on this.")
        
        with st.form("project_form"):
            title = st.text_input("Project Title", value=st.session_state.project_title, placeholder="e.g. Student Marks Prediction")
            desc = st.text_area("Problem Description", value=st.session_state.project_desc, placeholder="Describe what you want to predict...", height=150)
            
            st.write("---")
            st.write("**Dataset Availability**")
            data_source = st.radio("Choose option:", ["I have a dataset", "Suggest a dataset"], horizontal=True)
            
            submitted = st.form_submit_button("Save & Continue", type="primary")
            
            if submitted:
                if title and desc:
                    st.session_state.project_title = title
                    st.session_state.project_desc = desc
                    st.success("✅ Project initialized! Go to 'Data Setup' next.")
                    
                    if data_source == "Suggest a dataset":
                        st.session_state.recommendation_mode = True
                else:
                    st.error("Please fill in the details.")
    
    with col2:
        st.markdown("### 💡 How It Works")
        st.markdown("""
        <div style="background:white; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <b>1. Define Problem</b><br>
            Tell the system what you want to solve.<br><br>
            <b>2. Connect Data</b><br>
            Upload a CSV or get a recommendation.<br><br>
            <b>3. Auto-Train (PyCaret)</b><br>
            The system uses PyCaret to train 15+ models automatically.<br><br>
            <b>4. Get Results</b><br>
            View the leaderboard and download the report.
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.get('recommendation_mode'):
        st.markdown("---")
        st.subheader("🤖 AI Recommendations")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("🏠 Housing Data")
        with c2:
            st.button("🎓 Student Marks")
        with c3:
            st.button("🩺 Heart Health")

# --- PAGE 2: DATA SETUP ---
elif main_page == "Data Setup":
    st.markdown('<div class="gradient-header">📂 Data Setup</div>', unsafe_allow_html=True)
    
    if not st.session_state.project_title:
        st.warning("⚠️ Please define your project in 'Home' first.")
    else:
        st.markdown(f"**Project:** {st.session_state.project_title}")
        
        uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
        
        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset Uploaded!")
            
        if st.session_state.df is not None:
            st.markdown("---")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Rows", st.session_state.df.shape[0])
            m2.metric("Total Columns", st.session_state.df.shape[1])
            m3.metric("Missing Values", st.session_state.df.isnull().sum().sum())
            
            st.subheader("👀 Data Preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 Target Configuration")
            
            col_target, col_info = st.columns(2)
            
            with col_target:
                target = st.selectbox("Select Target Column", st.session_state.df.columns)
                st.session_state.target = target
            
            with col_info:
                if target:
                    task = detect_task(st.session_state.df, target)
                    st.session_state.task_type = task
                    st.info(f"Detected Task: **{task}**")
                    st.session_state.setup_done = True

# --- PAGE 3: AUTOML & RESULTS ---
elif main_page == "AutoML & Results":
    st.markdown('<div class="gradient-header">🚀 AutoML & Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.setup_done:
        st.warning("⚠️ Please upload data and select a target in 'Data Setup' first.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("⚙️ Training Configuration")
            st.write(f"Target: **{st.session_state.target}**")
            st.write(f"Task: **{st.session_state.task_type}**")
            
            if st.button("🚀 Start PyCaret AutoML", type="primary"):
                with st.spinner("🤖 PyCaret is analyzing your data... (Preprocessing -> Training -> Comparison)"):
                    leaderboard, best_model = run_pycaret_automl(
                        st.session_state.df, 
                        st.session_state.target, 
                        st.session_state.task_type
                    )
                    st.session_state.leaderboard = leaderboard
                    st.session_state.best_model = best_model
                    st.rerun()
                
        with col2:
            st.info("System will initialize PyCaret to train multiple models automatically.")
            
        if st.session_state.leaderboard is not None:
            st.markdown("---")
            st.subheader("🏆 Model Leaderboard")
            
            # Highlight best model
            st.dataframe(st.session_state.leaderboard.style.highlight_max(axis=0, color='#d1fae5'), use_container_width=True)
            
            best_model_name = str(st.session_state.best_model)
            st.success(f"🥇 Best Model Identified: **{best_model_name}**")
            
            st.markdown("### 📄 Final Report")
            
            # Helper to safely get accuracy or R2
            metric_col = st.session_state.leaderboard.columns[1] # Usually Accuracy or MAE
            best_score = st.session_state.leaderboard.iloc[0,1]
            
            report = f"""
            PROJECT REPORT: {st.session_state.project_title}
            ------------------------------------------------
            Problem: {st.session_state.project_desc}
            Dataset Rows: {st.session_state.df.shape[0]}
            Target: {st.session_state.target}
            Task: {st.session_state.task_type}
            
            WINNING MODEL: {best_model_name}
            SCORE ({metric_col}): {best_score:.4f}
            
            LEADERBOARD (Top 5):
            {st.session_state.leaderboard.head(5).to_string()}
            """
            st.download_button("Download Report", report, "report.txt")
