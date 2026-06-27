import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- DASHBOARD CONFIGURATION ---
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Custom Human-Designed Professional Styling (Clean, Minimal, Non-AI Template Look)
st.markdown("""
<style>
    /* Global font cleanup */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header { font-size: 28px; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
    .sub-header { font-size: 14px; color: #64748b; margin-bottom: 24px; }
    
    /* Clean white enterprise analytics cards */
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .metric-label { font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-num { font-size: 28px; font-weight: 700; color: #0f172a; margin-top: 4px; }
    
    /* Structural content layouts */
    .info-panel {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #334155;
        line-height: 1.5;
    }
    .info-panel strong { color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CSV DATA PIPELINE ---
st.sidebar.markdown("<h2 style='font-size: 18px; font-weight: 700; margin-bottom: 2px; color: #1e293b;'>Data Management</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 12px; color: #64748b; margin-top: 0; margin-bottom: 15px;'>Import spreadsheet records and run student simulations.</p>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload Student Records (CSV)", type=["csv"])

# Target configuration columns
feature_cols = [
    'age', 'gender', 'study_hours_per_day', 'attendance_percentage',
    'previous_gpa', 'assignments_completed', 'extracurricular_activities',
    'internet_access', 'family_income_level', 'parent_education_level',
    'sleep_hours', 'stress_level'
]

# State variables initialization
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'active_mode' not in st.session_state:
    st.session_state.active_mode = "Default Demo Mode"

# --- RUN COMPUTATION BUTTON LOGIC ---
if uploaded_file is not None:
    st.sidebar.markdown("---")
    st.sidebar.warning("New data loaded. Click below to analyze.")
    
    if st.sidebar.button("Run Dataset Analysis", type="primary", use_container_width=True):
        raw_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in feature_cols if c not in raw_df.columns]
        if missing_cols:
            st.error(f"Error: Missing expected columns: {missing_cols}")
            st.stop()
        
        st.session_state.dataset = raw_df
        st.session_state.pipeline_run = True
        st.session_state.active_mode = "Active CSV Dataset"
        st.sidebar.success("Analysis complete.")
else:
    if st.session_state.dataset is None or st.session_state.active_mode == "Active CSV Dataset":
        np.random.seed(42)
        n = 150
        st.session_state.dataset = pd.DataFrame({
            'student_id': range(1, n+1), 'age': np.random.randint(18,23,n), 'gender': np.random.choice([0,1],n),
            'study_hours_per_day': np.random.uniform(1, 8, n), 'attendance_percentage': np.random.randint(50,101,n),
            'previous_gpa': np.random.uniform(1.5, 4.0, n), 'assignments_completed': np.random.randint(40,101,n),
            'extracurricular_activities': np.random.choice([0,1],n), 'internet_access': np.random.choice([0,1],n),
            'family_income_level': np.random.randint(1,4,n), 'parent_education_level': np.random.randint(1,5,n),
            'sleep_hours': np.random.randint(5,10,n), 'stress_level': np.random.randint(1,6,n),
            'final_marks': np.random.randint(100, 400, n), 'passed': np.random.choice([0,1],n), 'dropped_out': np.random.choice([0,1],n)
        })
        st.session_state.pipeline_run = True
        st.session_state.active_mode = "Default Demo Mode"

df = st.session_state.dataset

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current View:** {st.session_state.active_mode}")
st.sidebar.markdown(f"**Total Sample Size:** {len(df)} records")

# --- SIDEBAR FEATURE CONTROLS ---
st.sidebar.markdown("<h3 style='font-size: 14px; font-weight: 700; margin-top: 20px; color: #1e293b;'>Student Profile Simulator</h3>", unsafe_allow_html=True)

sim_inputs = {}
sim_inputs['age'] = st.sidebar.slider("Age", int(df['age'].min()), max(int(df['age'].max()), int(df['age'].min())+1), int(df['age'].mean()))
sim_inputs['gender'] = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
sim_inputs['study_hours_per_day'] = st.sidebar.slider("Study Hours (Daily)", float(df['study_hours_per_day'].min()), max(float(df['study_hours_per_day'].max()), float(df['study_hours_per_day'].min())+1.0), float(df['study_hours_per_day'].mean()), 0.1)
sim_inputs['attendance_percentage'] = st.sidebar.slider("Attendance Rate (%)", int(df['attendance_percentage'].min()), max(int(df['attendance_percentage'].max()), int(df['attendance_percentage'].min())+1), int(df['attendance_percentage'].mean()))
sim_inputs['previous_gpa'] = st.sidebar.slider("Previous GPA Score", float(df['previous_gpa'].min()), max(float(df['previous_gpa'].max()), float(df['previous_gpa'].min())+0.5), float(df['previous_gpa'].mean()), 0.1)
sim_inputs['assignments_completed'] = st.sidebar.slider("Completed Assignments (%)", int(df['assignments_completed'].min()), max(int(df['assignments_completed'].max()), int(df['assignments_completed'].min())+1), int(df['assignments_completed'].mean()))
sim_inputs['extracurricular_activities'] = st.sidebar.selectbox("Extracurricular Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
sim_inputs['internet_access'] = st.sidebar.selectbox("Internet Access", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
sim_inputs['family_income_level'] = st.sidebar.slider("Income Level Category", int(df['family_income_level'].min()), max(int(df['family_income_level'].max()), int(df['family_income_level'].min())+1), int(df['family_income_level'].mean()))
sim_inputs['parent_education_level'] = st.sidebar.slider("Parent Education Tier", int(df['parent_education_level'].min()), max(int(df['parent_education_level'].max()), int(df['parent_education_level'].min())+1), int(df['parent_education_level'].mean()))
sim_inputs['sleep_hours'] = st.sidebar.slider("Sleep Duration (Hours)", int(df['sleep_hours'].min()), max(int(df['sleep_hours'].max()), int(df['sleep_hours'].min())+1), int(df['sleep_hours'].mean()))
sim_inputs['stress_level'] = st.sidebar.slider("Stress Level Scale", int(df['stress_level'].min()), max(int(df['stress_level'].max()), int(df['stress_level'].min())+1), int(df['stress_level'].mean()))

input_df = pd.DataFrame([sim_inputs])[feature_cols]

# Modeling matrix math calculations
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(df[feature_cols])
input_scaled = scaler.transform(input_df)

# --- MODEL PROCESSING ENGINES ---
has_final_marks = 'final_marks' in df.columns
if has_final_marks:
    lr = LinearRegression().fit(df[feature_cols], df['final_marks'])
    pred_marks = float(lr.predict(input_df)[0])
else:
    pred_marks = 0.0

has_passed = 'passed' in df.columns and len(df['passed'].unique()) > 1
if has_passed:
    logr = LogisticRegression().fit(X_scaled_full, df['passed'])
    prob_pass = float(logr.predict_proba(input_scaled)[0][1] * 100)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(df[feature_cols], df['passed'])
    sim_pred_class = "Passing Profile" if dt.predict(input_df)[0] == 1 else "At-Risk Profile"
else:
    prob_pass = 50.0
    sim_pred_class = "Insufficient data"

has_dropout = 'dropped_out' in df.columns and len(df['dropped_out'].unique()) > 1
if has_dropout:
    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(df[feature_cols], df['dropped_out'])
    prob_drop = float(rf.predict_proba(input_df)[0][1] * 100)
else:
    prob_drop = 0.0

knn = NearestNeighbors(n_neighbors=min(5, len(df))).fit(X_scaled_full)
km = KMeans(n_clusters=min(4, len(df)), random_state=42).fit(X_scaled_full)
pca = PCA(n_components=2).fit(X_scaled_full)
df['Cluster'] = km.labels_

# --- MAIN SCREEN INTERFACE ---
st.markdown("<div class='main-header'>Student Analytics & Retention Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A clean, metrics-driven overview of academic performance trends and risk indicators.</div>", unsafe_allow_html=True)

# Standard human style metric row
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    val_str = f"{pred_marks:.1f} pts" if has_final_marks else "N/A"
    st.markdown(f'<div class="metric-container"><div class="metric-label">Estimated Exam Score</div><div class="metric-num" style="color: #2563eb;">{val_str}</div></div>', unsafe_allow_html=True)
with m_col2:
    val_str = f"{prob_pass:.1f}%" if has_passed else "N/A"
    st.markdown(f'<div class="metric-container"><div class="metric-label">Passing Probability</div><div class="metric-num" style="color: #16a34a;">{val_str}</div></div>', unsafe_allow_html=True)
with m_col3:
    val_str = f"{prob_drop:.1f}%" if has_dropout else "N/A"
    st.markdown(f'<div class="metric-container"><div class="metric-label">Retention Risk Index</div><div class="metric-num" style="color: #dc2626;">{val_str}</div></div>', unsafe_allow_html=True)

# 7 standard navigation tab panes
tabs = st.tabs([
    "Grade Estimation", 
    "Passing Status Checklist", 
    "Similar Historical Records", 
    "Decision Rules", 
    "Retention Risks",
    "Dataset Clustering Mapping", 
    "Executive Overview Summary"
])

# TAB 1: LINEAR REGRESSION
with tabs[0]:
    st.markdown("### Grade Estimation Engine")
    st.markdown("<div class='info-panel'>Calculates estimated exam performance indicators by evaluating metric variations and determining feature impact scores based on your active dataset context.</div>", unsafe_allow_html=True)
    
    if has_final_marks:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(label="Calculated Target Marks Baseline", value=f"{pred_marks:.2f}")
        with c2:
            weights = pd.DataFrame({'Variable': feature_cols, 'Impact Score': lr.coef_}).sort_values(by='Impact Score')
            fig_lr = px.bar(weights, x='Impact Score', y='Variable', orientation='h', title="Feature Importance Metrics", color='Impact Score', color_continuous_scale='bluered')
            fig_lr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_lr, use_container_width=True)
    else:
        st.warning("Column 'final_marks' not found inside the active dataset.")

# TAB 2: LOGISTIC REGRESSION
with tabs[1]:
    st.markdown("### Passing Status Checklist")
    st.markdown("<div class='info-panel'>Calculates the likelihood of a student meeting baseline passing thresholds based on your custom configuration inputs.</div>", unsafe_allow_html=True)
    
    if has_passed:
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob_pass, title={'text': "Passing Likelihood Gauge"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#16a34a"}}))
            fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col_l2:
            if has_final_marks:
                fig_box = px.box(df, x='passed', y='final_marks', color='passed', title="Exam Score Distribution by Pass/Fail Status", color_discrete_sequence=["#dc2626", "#16a34a"])
                fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("Column 'passed' not found inside the active dataset.")

# TAB 3: KNN
with tabs[2]:
    st.markdown("### Similar Historical Records Finder")
    st.markdown("<div class='info-panel'>Queries the active student database database to pull the 5 historical rows that match closest to your sidebar simulation metrics.</div>", unsafe_allow_html=True)
    
    distances, indices = knn.kneighbors(input_scaled)
    matched_records = df.iloc[indices[0]].copy()
    
    st.markdown("#### Nearest Matched Student Records")
    show_cols = ['student_id'] + [c for c in feature_cols if c in df.columns]
    if has_final_marks: show_cols.append('final_marks')
    st.dataframe(matched_records[show_cols], use_container_width=True)
    
    if has_final_marks and has_passed:
        fig_knn = px.scatter(df, x='study_hours_per_day', y='final_marks', color='passed', title="Dataset Matrix Map with Highlighted Sim Student Location", opacity=0.3, color_discrete_sequence=["#dc2626", "#16a34a"])
        fig_knn.add_trace(go.Scatter(x=matched_records['study_hours_per_day'], y=matched_records['final_marks'], mode='markers', marker=dict(size=12, color='#eab308', symbol='diamond', line=dict(width=1, color='black')), name='Matched Profiles'))
        fig_knn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_knn, use_container_width=True)

# TAB 4: DECISION TREE
with tabs[3]:
    st.markdown("### Segment Performance Split Criteria")
    st.markdown("<div class='info-panel'>Calculates data segmentation patterns to show which key baseline metrics split high-performing students from struggling cohorts.</div>", unsafe_allow_html=True)
    
    if has_passed:
        dt_features = pd.DataFrame({'Variable': feature_cols, 'Gini Importance': dt.feature_importances_}).sort_values(by='Gini Importance', ascending=False)
        c_dt1, c_dt2 = st.columns([1, 2])
        with c_dt1:
            st.metric("Model Segment Classification", sim_pred_class)
        with c_dt2:
            fig_dt = px.bar(dt_features, x='Gini Importance', y='Variable', orientation='h', title="Primary Feature Split Tree Weights", color='Gini Importance', color_continuous_scale='gray')
            fig_dt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_dt, use_container_width=True)
    else:
        st.warning("Column 'passed' required for split visualization calculations.")

# TAB 5: RANDOM FOREST
with tabs[4]:
    st.markdown("### Attrition Risk Evaluation Matrix")
    st.markdown("<div class='info-panel'>Calculates dropout risks by running aggregated criteria splits across historical student profiles.</div>", unsafe_allow_html=True)
    
    if has_dropout:
        rf_features = pd.DataFrame({'Variable': feature_cols, 'Importance Rating': rf.feature_importances_}).sort_values(by='Importance Rating', ascending=False)
        c_rf1, c_rf2 = st.columns(2)
        with c_rf1:
            fig_rf_gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob_drop, title={'text': "Aggregated Dropout Danger Index"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#dc2626"}}))
            fig_rf_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_rf_gauge, use_container_width=True)
        with c_rf2:
            fig_rf_bar = px.bar(rf_features, x='Importance Rating', y='Variable', orientation='h', title="Risk Factors Weight Variables", color='Importance Rating', color_continuous_scale='reds')
            fig_rf_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_rf_bar, use_container_width=True)
    else:
        st.warning("Column 'dropped_out' missing inside data configuration limits.")

# TAB 6: K-MEANS & PCA
with tabs[5]:
    st.markdown("### Profile Archetypes & Clustering Map")
    st.markdown("<div class='info-panel'>Segments the student population into 4 behavioral archetypes based on features, projecting them onto a clean scatter chart view.</div>", unsafe_allow_html=True)
    
    pca_coordinates = pca.transform(X_scaled_full)
    pca_df = pd.DataFrame(pca_coordinates, columns=['PC 1', 'PC 2'])
    pca_df['Student Group'] = df['Cluster'].map(lambda x: f"Group Cohort {x+1}")
    
    if has_final_marks:
        pca_df['Marks Index'] = df['final_marks']
        fig_pca = px.scatter(pca_df, x='PC 1', y='PC 2', color='Student Group', size='Marks Index', title="Population Clusters Projection View", color_discrete_sequence=px.colors.qualitative.Safe)
    else:
        fig_pca = px.scatter(pca_df, x='PC 1', y='PC 2', color='Student Group', title="Population Clusters Projection View", color_discrete_sequence=px.colors.qualitative.Safe)
    
    sim_pca = pca.transform(input_scaled)
    fig_pca.add_trace(go.Scatter(x=[sim_pca[0][0]], y=[sim_pca[0][1]], mode='markers', marker=dict(size=14, color='#ef4444', symbol='x', line=dict(width=1, color='black')), name='Simulated Point Location'))
    fig_pca.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pca, use_container_width=True)

# TAB 7: EXECUTIVE OVERVIEW SUMMARY
with tabs[6]:
    st.markdown("### Performance Overview Report Summary")
    st.markdown("<div class='info-panel'>Generates a structured, concise breakdown of the simulated profile metrics against current dataset parameters.</div>", unsafe_allow_html=True)
    
    status_text = "Stable Academic Status Profile" if prob_drop < 30 and pred_marks > 200 else "Needs Academic/Retention Review"
    
    st.markdown(f"""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);">
        <h4 style="margin-top: 0; color: #0f172a; font-weight: 700; font-size: 16px;">System Evaluation Breakdown</h4>
        <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 15px 0;">
        <p style="font-size: 14px; color: #334155;"><strong>Student Review Status:</strong> {status_text}</p>
        <ul style="font-size: 14px; color: #475569; margin-top: 10px; line-height: 1.6;">
            <li><strong>Expected Score Projection:</strong> Baseline performance metrics track at roughly <strong>{pred_marks:.1f} marks</strong> based on active variable weights.</li>
            <li><strong>Requirements Safety Limit:</strong> The configured setup holds a <strong>{prob_pass:.1f}% probability</strong> of meeting course passing requirements.</li>
            <li><strong>Retention Assessment Risk:</strong> Modeled criteria indicate a <strong>{prob_drop:.1f}% attrition risk factor</strong> based on historical profiles.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)