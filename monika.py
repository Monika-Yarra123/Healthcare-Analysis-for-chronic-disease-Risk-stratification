import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Healthcare Analysis for Chronic Disease Risk Stratification", layout="wide")
st.title("🏥 Healthcare Analysis for Chronic Disease Risk Stratification")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    file_path = r"C:\Users\Monika Yarra\Downloads\healthcare analysis.csv"
    df = pd.read_csv(file_path)
    
    

# ===============================
# CREATE DIABETES RISK BIN & FLAG
# ===============================
    df['Diabetes_Risk_Bin'] = pd.cut(
    df['diabetes_risk_score'],
    bins=[-0.01, 0.33, 0.66, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# FORCE RECALCULATION - Always create based on risk score threshold
    df['has_diabetes'] = (df['diabetes_risk_score'] >= 0.66).astype(int)
    
    # ===============================
    # CREATE CARDIO RISK SCORE & BIN
    # ===============================
    bmi_risk = np.where(df['BMI'] < 18.5, 0.1,
                np.where(df['BMI'] < 25, 0.2,
                np.where(df['BMI'] < 30, 0.5,
                np.where(df['BMI'] < 35, 0.7,
                np.where(df['BMI'] < 40, 0.85, 0.95)))))
    
    sbp_risk = np.where(df['Systolic Blood Pressure'] < 120, 0.2,
                   np.where(df['Systolic Blood Pressure'] < 130, 0.4,
                   np.where(df['Systolic Blood Pressure'] < 140, 0.6,
                   np.where(df['Systolic Blood Pressure'] < 160, 0.8, 0.95))))
    
    dbp_risk = np.where(df['Diastolic Blood Pressure'] < 80, 0.2,
                   np.where(df['Diastolic Blood Pressure'] < 90, 0.5,
                   np.where(df['Diastolic Blood Pressure'] < 100, 0.75, 0.9)))
    
    df['Cardio_Risk_Score'] = np.clip(
        0.25 * bmi_risk + 0.40 * sbp_risk + 0.35 * dbp_risk, 0, 1
    )
    
    df['CVD_Risk_Bin'] = pd.cut(
        df['Cardio_Risk_Score'],
        bins=[-0.01, 0.40, 0.65, 1.01],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # FIXED: Use consistent threshold (>= 0.65) like diabetes
    df['has_cardio'] = (df['Cardio_Risk_Score'] >= 0.65).astype(int)
    
    # ===============================
    # CREATE COPD RISK SCORE & BIN
    # ===============================
    age_risk = np.where(df['AGE'] < 40, 0.1,
                   np.where(df['AGE'] < 50, 0.3,
                   np.where(df['AGE'] < 60, 0.5,
                   np.where(df['AGE'] < 70, 0.7, 0.9))))
    
    # FIXED: Proper handling of optional columns
    if 'risk_smoking_tobacco' in df.columns:
        smoking_risk = df['risk_smoking_tobacco'].fillna(0.5)
    else:
        smoking_risk = np.full(len(df), 0.5)
    
    if 'risk_genetic' in df.columns:
        genetic_risk = df['risk_genetic'].fillna(0.3)
    else:
        genetic_risk = np.full(len(df), 0.3)
    
    if 'risk_respiratory_infection' in df.columns:
        respiratory_risk = df['risk_respiratory_infection'].fillna(0.4)
    else:
        respiratory_risk = np.full(len(df), 0.4)
    
    df['COPD_Risk_Score'] = np.clip(
        0.30 * age_risk + 0.35 * smoking_risk + 0.20 * genetic_risk + 0.15 * respiratory_risk, 0, 1
    )
    
    df['COPD_Risk_Bin'] = pd.cut(
        df['COPD_Risk_Score'],
        bins=[-0.01, 0.40, 0.65, 1.01],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # FIXED: Use consistent threshold (>= 0.65) for all diseases
    df['has_copd'] = (df['COPD_Risk_Score'] >= 0.65).astype(int)
    
    # Additional columns
    df['Age_Group'] = pd.cut(df['AGE'], bins=[0, 18, 30, 45, 60, 120], 
                              labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], 
                                 labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    return df

# Load data
df = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔍 Global Filters")
age_min, age_max = int(df['AGE'].min()), int(df['AGE'].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
gender_options = df['GENDER'].unique().tolist()
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
county_options = sorted(df['COUNTY'].dropna().unique().tolist())
selected_counties = st.sidebar.multiselect("County", county_options, default=county_options)

filtered_df = df[
    (df['AGE'] >= age_range[0]) & 
    (df['AGE'] <= age_range[1]) &
    (df['GENDER'].isin(selected_gender)) &
    (df['COUNTY'].isin(selected_counties))
].copy()

st.sidebar.success(f"✅ {len(filtered_df)} patients")
st.sidebar.info(f"📊 Out of {len(df)} total")

# -------------------------------
# Tabs
# -------------------------------
overview_tab, diabetes_tab, cvd_tab, copd_tab, x_tab, checking = st.tabs([
    "📊 Overview", "🩺 Diabetes", "🫀 CVD", "🫁 COPD", "📋 Comorbidity", "🔍 Individual Patient Risk"
])

# ===============================
# TAB 1: OVERVIEW
# ===============================
with overview_tab:
    st.header("📊 Overview")
    st.info(f"🔍 Showing {len(filtered_df)} of {len(df)} total patients")
    st.markdown("---")
    
    # ===============================
    # 4 KPIs
    # ===============================
    col1, col2, col3, col4 = st.columns(4)
    
    total_patients = len(filtered_df)
    high_risk_diabetes = len(filtered_df[filtered_df['Diabetes_Risk_Bin'] == 'High Risk'])
    high_risk_cvd = len(filtered_df[filtered_df['CVD_Risk_Bin'] == 'High Risk'])
    high_risk_copd = len(filtered_df[filtered_df['COPD_Risk_Bin'] == 'High Risk'])
    
    col1.metric("Total Patients", total_patients)
    col2.metric("🩺 High Risk Diabetes", high_risk_diabetes, 
                delta=f"{(high_risk_diabetes/total_patients*100):.1f}%")
    col3.metric("🫀 High Risk CVD", high_risk_cvd, 
                delta=f"{(high_risk_cvd/total_patients*100):.1f}%")
    col4.metric("🫁 High Risk COPD", high_risk_copd, 
                delta=f"{(high_risk_copd/total_patients*100):.1f}%")
    
    st.markdown("---")
    
    # ===============================
    # Bar Graph: Disease Risk Distribution
    # ===============================
    st.subheader("📊 Disease Risk Distribution Comparison")
    
    # ADDED: Filter for diseases to display
    diseases_to_show = st.multiselect(
        "Select Diseases to Display:",
        options=['Diabetes', 'CVD', 'COPD'],
        default=['Diabetes', 'CVD', 'COPD'],
        key='disease_dist_filter'
    )
    
    # Prepare data for selected diseases only
    all_disease_data = []
    
    if 'Diabetes' in diseases_to_show:
        diabetes_counts = filtered_df['Diabetes_Risk_Bin'].value_counts().reset_index()
        diabetes_counts.columns = ['Risk Level', 'Count']
        diabetes_counts['Disease'] = 'Diabetes'
        all_disease_data.append(diabetes_counts)
    
    if 'CVD' in diseases_to_show:
        cvd_counts = filtered_df['CVD_Risk_Bin'].value_counts().reset_index()
        cvd_counts.columns = ['Risk Level', 'Count']
        cvd_counts['Disease'] = 'CVD'
        all_disease_data.append(cvd_counts)
    
    if 'COPD' in diseases_to_show:
        copd_counts = filtered_df['COPD_Risk_Bin'].value_counts().reset_index()
        copd_counts.columns = ['Risk Level', 'Count']
        copd_counts['Disease'] = 'COPD'
        all_disease_data.append(copd_counts)
    
    if all_disease_data:
        combined_risk = pd.concat(all_disease_data)
        
        fig = px.bar(
            combined_risk, 
            x='Risk Level', 
            y='Count', 
            color='Disease',
            barmode='group',
            text='Count',
            color_discrete_map={'Diabetes': '#e74c3c', 'CVD': '#3498db', 'COPD': '#f39c12'},
            title='Risk Level Distribution Across Selected Diseases',
            category_orders={'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk']}
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please select at least one disease to display.")
    
    st.markdown("---")
    
    # ===============================
    # Line Chart: Predictive Risk Score Trends
    # ===============================
    st.subheader("📈 Predictive Risk Score Trends Over Years")
    
    diseases_trend_filter = st.multiselect(
        "Select Diseases to Display:",
        options=['Diabetes', 'CVD', 'COPD'],
        default=['Diabetes', 'CVD', 'COPD'],
        key='trend_disease_filter'
    )
    
    if 'Year' in filtered_df.columns and len(diseases_trend_filter) > 0:
        trend_data_list = []
        
        if 'Diabetes' in diseases_trend_filter:
            diabetes_by_year = filtered_df.groupby('Year')['diabetes_risk_score'].mean().reset_index()
            diabetes_by_year.columns = ['Year', 'Risk Score']
            diabetes_by_year['Disease'] = 'Diabetes'
            trend_data_list.append(diabetes_by_year)
        
        if 'CVD' in diseases_trend_filter:
            cvd_by_year = filtered_df.groupby('Year')['Cardio_Risk_Score'].mean().reset_index()
            cvd_by_year.columns = ['Year', 'Risk Score']
            cvd_by_year['Disease'] = 'CVD'
            trend_data_list.append(cvd_by_year)
        
        if 'COPD' in diseases_trend_filter:
            copd_by_year = filtered_df.groupby('Year')['COPD_Risk_Score'].mean().reset_index()
            copd_by_year.columns = ['Year', 'Risk Score']
            copd_by_year['Disease'] = 'COPD'
            trend_data_list.append(copd_by_year)
        
        combined_trends = pd.concat(trend_data_list)
        fig_trends = px.line(
            combined_trends, 
            x='Year', 
            y='Risk Score', 
            color='Disease',
            markers=True,
            color_discrete_map={'Diabetes': '#e74c3c', 'CVD': '#3498db', 'COPD': '#f39c12'},
            title='Average Risk Score Trends by Year'
        )
        fig_trends.update_layout(yaxis_title='Average Risk Score (0-1)')
        st.plotly_chart(fig_trends, use_container_width=True)
    
    elif 'Year' not in filtered_df.columns:
        st.warning("'Year' column not found in dataset")
    else:
        st.warning("⚠️ Please select at least one disease to display.")
    
    st.markdown("---")

    # ===============================
    # Age Distribution by Disease (Single Plot)
    # ===============================
    st.subheader("👥 Age Distribution by Disease")

    diseases_age_filter = st.multiselect(
        "Select Diseases to Display:",
        options=['Diabetes', 'CVD', 'COPD'],
        default=['Diabetes', 'CVD', 'COPD'],
        key='age_disease_filter'
    )

    age_data_list = []

    if 'Diabetes' in diseases_age_filter:
        diabetes_age = filtered_df[filtered_df['Diabetes_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]
        diabetes_age = diabetes_age['AGE'].value_counts().sort_index().reset_index()
        diabetes_age.columns = ['Age', 'Count']
        diabetes_age['Disease'] = 'Diabetes'
        age_data_list.append(diabetes_age)

    if 'CVD' in diseases_age_filter:
        cvd_age = filtered_df[filtered_df['CVD_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]
        cvd_age = cvd_age['AGE'].value_counts().sort_index().reset_index()
        cvd_age.columns = ['Age', 'Count']
        cvd_age['Disease'] = 'CVD'
        age_data_list.append(cvd_age)

    if 'COPD' in diseases_age_filter:
        copd_age = filtered_df[filtered_df['COPD_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]
        copd_age = copd_age['AGE'].value_counts().sort_index().reset_index()
        copd_age.columns = ['Age', 'Count']
        copd_age['Disease'] = 'COPD'
        age_data_list.append(copd_age)

    if age_data_list:
        combined_age_data = pd.concat(age_data_list)
        fig_age_combined = px.line(
            combined_age_data,
            x='Age',
            y='Count',
            color='Disease',
            markers=True,
            color_discrete_map={'Diabetes': '#e74c3c', 'CVD': '#3498db', 'COPD': '#f39c12'},
            title="Age Distribution of Medium/High Risk Patients"
        )
        fig_age_combined.update_layout(yaxis_title='Number of Patients')
        st.plotly_chart(fig_age_combined, use_container_width=True)
    else:
        st.warning("⚠️ Please select at least one disease to display.")

    st.markdown("---")

    # ===============================
    # Gender Distribution by Disease
    # ===============================
    st.subheader("⚧ Gender Distribution by Disease")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("🩺 Diabetes Patients")
        # FIXED: Show only patients who HAVE diabetes
        diabetes_gender = filtered_df[filtered_df['has_diabetes'] == 1]['GENDER'].value_counts()
        if len(diabetes_gender) > 0:
            fig_gender = px.pie(values=diabetes_gender.values, names=diabetes_gender.index, 
                               color_discrete_sequence=['#e74c3c', '#c0392b'])
            fig_gender.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.info("No diabetic patients in filtered data")
    
    with col2:
        st.markdown("🫀 CVD Patients")
        # FIXED: Show only patients who HAVE CVD
        cvd_gender = filtered_df[filtered_df['has_cardio'] == 1]['GENDER'].value_counts()
        if len(cvd_gender) > 0:
            fig_gender = px.pie(values=cvd_gender.values, names=cvd_gender.index, 
                               color_discrete_sequence=['#3498db', '#2980b9'])
            fig_gender.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.info("No CVD patients in filtered data")
    
    with col3:
        st.markdown("🫁 COPD Patients")
        # FIXED: Show only patients who HAVE COPD
        copd_gender = filtered_df[filtered_df['has_copd'] == 1]['GENDER'].value_counts()
        if len(copd_gender) > 0:
            fig_gender = px.pie(values=copd_gender.values, names=copd_gender.index, 
                               color_discrete_sequence=['#f39c12', '#e67e22'])
            fig_gender.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.info("No COPD patients in filtered data")
    
    st.markdown("---")
    
    # ===============================
    # Top 10 Counties by Disease Risk
    # ===============================
    st.subheader("🏆 Top 10 Counties by Disease Risk")
    
    county_disease = st.radio("Select Disease:", ['Diabetes', 'CVD', 'COPD', 'All Diseases'], horizontal=True)
    
    if county_disease == 'Diabetes':
        top_counties = filtered_df[filtered_df['Diabetes_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]['COUNTY'].value_counts().head(10).reset_index()
        top_counties.columns = ['County', 'Diabetes Risk Patients']
    
    elif county_disease == 'CVD':
        top_counties = filtered_df[filtered_df['CVD_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]['COUNTY'].value_counts().head(10).reset_index()
        top_counties.columns = ['County', 'CVD Risk Patients']
    
    elif county_disease == 'COPD':
        top_counties = filtered_df[filtered_df['COPD_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]['COUNTY'].value_counts().head(10).reset_index()
        top_counties.columns = ['County', 'COPD Risk Patients']
    
    else:
        filtered_df['High_Risk_Count'] = (
            (filtered_df['Diabetes_Risk_Bin'].isin(['Medium Risk','High Risk'])).astype(int) +
            (filtered_df['CVD_Risk_Bin'].isin(['Medium Risk','High Risk'])).astype(int) +
            (filtered_df['COPD_Risk_Bin'].isin(['Medium Risk','High Risk'])).astype(int)
        )
        top_counties = filtered_df.groupby('COUNTY')['High_Risk_Count'].sum().sort_values(ascending=False).head(10).reset_index()
        top_counties.columns = ['County', 'High Risk Patients (All Diseases)']
    
    top_counties.insert(0, 'Rank', range(1, len(top_counties) + 1))
    st.dataframe(top_counties, use_container_width=True, hide_index=True)

# ===============================
# TAB 2: DIABETES ANALYSIS
# ===============================
with diabetes_tab:
    st.header("🩺 Diabetes Analysis")
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filter Patients")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_filter = st.multiselect("Gender", df['GENDER'].unique().tolist(), 
                                       default=df['GENDER'].unique().tolist(), key='diab_gender')
    with col2:
        county_filter = st.multiselect("County", sorted(df['COUNTY'].unique().tolist()),
                                       default=sorted(df['COUNTY'].unique().tolist())[:10], key='diab_county')
    with col3:
        risk_filter = st.multiselect("Risk Bin", ['Low Risk', 'Medium Risk', 'High Risk'],
                                     default=['Low Risk', 'Medium Risk', 'High Risk'], key='diab_risk')
    
    diabetes_filtered = df[
        (df['GENDER'].isin(gender_filter)) &
        (df['COUNTY'].isin(county_filter)) &
        (df['Diabetes_Risk_Bin'].isin(risk_filter))
    ]
    
    st.info(f"📊 Showing {len(diabetes_filtered)} patients")
    st.markdown("---")
    
    # Table
    st.subheader("Patient Risk Details")
    diab_table = diabetes_filtered[['Id', 'diabetes_risk_score', 'Diabetes_Risk_Bin', 'has_diabetes', 'GENDER', 'COUNTY']].copy()
    diab_table.columns = ['ID', 'Risk Score', 'Risk Bin', 'Diabetes Status', 'Gender', 'County']
    diab_table['Diabetes Status'] = diab_table['Diabetes Status'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
    st.dataframe(diab_table, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # FIXED: Show risk distribution only for patients WHO HAVE diabetes
    st.subheader("Diabetes Patients by Risk Level")
    diabetic_only = diabetes_filtered[diabetes_filtered['has_diabetes'] == 1]
    
    if len(diabetic_only) > 0:
        risk_counts = diabetic_only['Diabetes_Risk_Bin'].value_counts().reset_index()
        risk_counts.columns = ['Risk Bin', 'Count']
        risk_counts = risk_counts.sort_values('Risk Bin', key=lambda x: x.map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}))
        
        fig = px.bar(risk_counts, x='Risk Bin', y='Count', text='Count', color='Risk Bin',
                     color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f1c40f', 'High Risk': '#e74c3c'})
        fig.update_traces(textposition='outside')
        fig.update_layout(title='Risk Level Distribution (Diabetic Patients Only)')
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Total Diabetic Patients: {len(diabetic_only):,}")
    else:
        st.warning("No diabetic patients (has_diabetes=1) in filtered data")
    
    st.markdown("---")
    
    # FIXED: Gender distribution only for patients WHO HAVE diabetes
    st.subheader("Gender Distribution")
    gender_data = diabetic_only['GENDER'].value_counts().reset_index()
    gender_data.columns = ['Gender', 'Count']
    
    if len(gender_data) > 0:
        fig = px.bar(gender_data, x='Gender', y='Count', text='Count', color='Gender',
                     color_discrete_map={'M': '#e74c3c', 'F': '#f39c12'})
        fig.update_traces(textposition='inside')
        fig.update_layout(title='Gender Distribution (Diabetic Patients Only)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No diabetic patients to display gender distribution")
    
    st.markdown("---")

    # Top counties
    top_diabetes_counties = (
        filtered_df[filtered_df['Diabetes_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]
        ['COUNTY']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_diabetes_counties.columns = ['County', 'High/Medium Risk Diabetes Patients']
    top_diabetes_counties.insert(0, 'Rank', range(1, len(top_diabetes_counties) + 1))

    st.subheader("🏆 Top 10 Counties by Diabetes Risk")
    st.dataframe(top_diabetes_counties, use_container_width=True, hide_index=True)

# ===============================
# TAB 3: CVD ANALYSIS
# ===============================
with cvd_tab:
    st.header("🫀 CVD Analysis")
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filter Patients")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_filter = st.multiselect("Gender", df['GENDER'].unique().tolist(),
                                       default=df['GENDER'].unique().tolist(), key='cvd_gender')
    with col2:
        county_filter = st.multiselect("County", sorted(df['COUNTY'].unique().tolist()),
                                       default=sorted(df['COUNTY'].unique().tolist())[:10], key='cvd_county')
    with col3:
        risk_filter = st.multiselect("Risk Bin", ['Low Risk', 'Medium Risk', 'High Risk'],
                                     default=['Low Risk', 'Medium Risk', 'High Risk'], key='cvd_risk')
    
    cvd_filtered = df[
        (df['GENDER'].isin(gender_filter)) &
        (df['COUNTY'].isin(county_filter)) &
        (df['CVD_Risk_Bin'].isin(risk_filter))
    ]
    
    st.info(f"📊 Showing {len(cvd_filtered)} patients")
    st.markdown("---")
    
    # Table
    st.subheader("Patient Risk Details")
    cvd_table = cvd_filtered[['Id', 'Cardio_Risk_Score', 'CVD_Risk_Bin', 'has_cardio', 'GENDER', 'COUNTY']].copy()
    cvd_table.columns = ['ID', 'Risk Score', 'Risk Bin', 'CVD Status', 'Gender', 'County']
    cvd_table['CVD Status'] = cvd_table['CVD Status'].map({0: 'No CVD', 1: 'Has CVD'})
    st.dataframe(cvd_table, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # FIXED: Show risk distribution only for patients WHO HAVE CVD
    st.subheader("CVD Patients by Risk Level")
    cvd_only = cvd_filtered[cvd_filtered['has_cardio'] == 1]
    
    if len(cvd_only) > 0:
        risk_counts = cvd_only['CVD_Risk_Bin'].value_counts().reset_index()
        risk_counts.columns = ['Risk Bin', 'Count']
        risk_counts = risk_counts.sort_values('Risk Bin', key=lambda x: x.map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}))
        
        fig = px.bar(risk_counts, x='Risk Bin', y='Count', text='Count', color='Risk Bin',
                     color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f1c40f', 'High Risk': '#e74c3c'})
        fig.update_traces(textposition='outside')
        fig.update_layout(title='Risk Level Distribution (CVD Patients Only)')
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Total CVD Patients: {len(cvd_only):,}")
    else:
        st.warning("No CVD patients (has_cardio=1) in filtered data")
    
    st.markdown("---")
    
    # FIXED: Gender distribution only for patients WHO HAVE CVD
    st.subheader("Gender Distribution")
    gender_data = cvd_only['GENDER'].value_counts().reset_index()
    gender_data.columns = ['Gender', 'Count']
    
    if len(gender_data) > 0:
        fig = px.bar(gender_data, x='Gender', y='Count', text='Count', color='Gender',
                     color_discrete_map={'M': '#3498db', 'F': '#9b59b6'})
        fig.update_traces(textposition='inside')
        fig.update_layout(title='Gender Distribution (CVD Patients Only)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No CVD patients to display gender distribution")
    
    st.markdown("---")

    # Top counties
    top_cardio_counties = (
        filtered_df[filtered_df['CVD_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]
        ['COUNTY']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_cardio_counties.columns = ['County', 'High/Medium Risk CVD Patients']
    top_cardio_counties.insert(0, 'Rank', range(1, len(top_cardio_counties) + 1))
    
    st.subheader("🏆 Top 10 Counties by CVD Risk")
    st.dataframe(top_cardio_counties, use_container_width=True, hide_index=True)

# ===============================
# TAB 4: COPD ANALYSIS
# ===============================
with copd_tab:
    st.header("🫁 COPD Analysis")
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filter Patients")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_filter = st.multiselect("Gender", df['GENDER'].unique().tolist(),
                                       default=df['GENDER'].unique().tolist(), key='copd_gender')
    with col2:
        county_filter = st.multiselect("County", sorted(df['COUNTY'].unique().tolist()),
                                       default=sorted(df['COUNTY'].unique().tolist())[:10], key='copd_county')
    with col3:
        risk_filter = st.multiselect("Risk Bin", ['Low Risk', 'Medium Risk', 'High Risk'],
                                     default=['Low Risk', 'Medium Risk', 'High Risk'], key='copd_risk')
    
    copd_filtered = df[
        (df['GENDER'].isin(gender_filter)) &
        (df['COUNTY'].isin(county_filter)) &
        (df['COPD_Risk_Bin'].isin(risk_filter))
    ]
    
    st.info(f"📊 Showing {len(copd_filtered)} patients")
    st.markdown("---")
    
    # Table
    st.subheader("Patient Risk Details")
    copd_table = copd_filtered[['Id', 'COPD_Risk_Score', 'COPD_Risk_Bin', 'has_copd', 'AGE', 'GENDER', 'COUNTY']].copy()
    copd_table.columns = ['ID', 'Risk Score', 'Risk Bin', 'COPD Status', 'Age', 'Gender', 'County']
    copd_table['COPD Status'] = copd_table['COPD Status'].map({0: 'No COPD', 1: 'Has COPD'})
    st.dataframe(copd_table, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # FIXED: Show risk distribution only for patients WHO HAVE COPD
    st.subheader("COPD Patients by Risk Level")
    copd_only = copd_filtered[copd_filtered['has_copd'] == 1]
    
    if len(copd_only) > 0:
        risk_counts = copd_only['COPD_Risk_Bin'].value_counts().reset_index()
        risk_counts.columns = ['Risk Bin', 'Count']
        risk_counts = risk_counts.sort_values('Risk Bin', key=lambda x: x.map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}))
        
        fig = px.bar(risk_counts, x='Risk Bin', y='Count', text='Count', color='Risk Bin',
                     color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f1c40f', 'High Risk': '#e74c3c'})
        fig.update_traces(textposition='outside')
        fig.update_layout(title='Risk Level Distribution (COPD Patients Only)')
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Total COPD Patients: {len(copd_only):,}")
    else:
        st.warning("No COPD patients (has_copd=1) in filtered data")
    
    st.markdown("---")
    
    # FIXED: Gender distribution only for patients WHO HAVE COPD
    st.subheader("Gender Distribution")
    gender_data = copd_only['GENDER'].value_counts().reset_index()
    gender_data.columns = ['Gender', 'Count']
    
    if len(gender_data) > 0:
        fig = px.bar(gender_data, x='Gender', y='Count', text='Count', color='Gender',
                     color_discrete_map={'M': '#f39c12', 'F': '#e67e22'})
        fig.update_traces(textposition='inside')
        fig.update_layout(title='Gender Distribution (COPD Patients Only)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No COPD patients to display gender distribution")
    
    st.markdown("---")

    # Top counties
    top_copd_counties = (
        filtered_df[filtered_df['COPD_Risk_Bin'].isin(['Medium Risk', 'High Risk'])]
        ['COUNTY']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_copd_counties.columns = ['County', 'High/Medium Risk COPD Patients']
    top_copd_counties.insert(0, 'Rank', range(1, len(top_copd_counties) + 1))
    
    st.subheader("🏆 Top 10 Counties by COPD Risk")
    st.dataframe(top_copd_counties, use_container_width=True, hide_index=True)

# ===============================
# TAB 5: X TAB - COMORBIDITY ANALYSIS
# ===============================
with x_tab:
    st.header("📋 Disease Comorbidity Analysis")
    st.markdown("### Understanding Disease Interlinking & Co-occurrence")
    st.markdown("---")

    # ----------------------------------------
    # Validate data availability
    # ----------------------------------------
    has_diabetes_count = len(filtered_df[filtered_df['has_diabetes'] == 1])
    has_cvd_count = len(filtered_df[filtered_df['has_cardio'] == 1])
    has_copd_count = len(filtered_df[filtered_df['has_copd'] == 1])

    min_patients_threshold = 10

    if has_diabetes_count >= min_patients_threshold and has_cvd_count >= min_patients_threshold and has_copd_count >= min_patients_threshold:
        st.success("✅ Sufficient data for **3-Disease Comorbidity Analysis**")
        analysis_mode = "three_diseases"
    elif has_diabetes_count >= min_patients_threshold and has_cvd_count >= min_patients_threshold:
        st.warning("⚠️ Limited COPD data — analyzing **Diabetes & CVD** only.")
        analysis_mode = "two_diseases"
    else:
        st.error("❌ Insufficient data for comorbidity analysis")
        analysis_mode = "insufficient"

    # ----------------------------------------
    # 3-DISEASE ANALYSIS
    # ----------------------------------------
    if analysis_mode == "three_diseases":
        st.markdown("---")
        
        # Create disease combinations
        filtered_df['Disease_Combo'] = (
            filtered_df['has_diabetes'].astype(str) + '-' +
            filtered_df['has_cardio'].astype(str) + '-' +
            filtered_df['has_copd'].astype(str)
        )

        combo_labels = {
            '0-0-0': 'No Disease',
            '1-0-0': 'Diabetes Only',
            '0-1-0': 'CVD Only',
            '0-0-1': 'COPD Only',
            '1-1-0': 'Diabetes + CVD',
            '1-0-1': 'Diabetes + COPD',
            '0-1-1': 'CVD + COPD',
            '1-1-1': 'All Three Diseases'
        }

        filtered_df['Disease_Label'] = filtered_df['Disease_Combo'].map(combo_labels)

        # Count each combination
        combo_counts = filtered_df['Disease_Label'].value_counts().reset_index()
        combo_counts.columns = ['Disease Combination', 'Patient Count']
        combo_counts.insert(0, 'Rank', range(1, len(combo_counts) + 1))

        # ----------------------------------------
        # Summary Table
        # ----------------------------------------
        st.subheader("📋 Three-Disease Comorbidity Summary")
        st.dataframe(combo_counts, use_container_width=True, hide_index=True, height=350)
        
        st.markdown("---")

        # ----------------------------------------
        # Pie Chart Only
        # ----------------------------------------
        st.subheader("🥧 Disease Co-occurrence Distribution")
        combo_with_disease = combo_counts[combo_counts['Disease Combination'] != 'No Disease']

        fig_pie = px.pie(
            combo_with_disease,
            values='Patient Count',
            names='Disease Combination',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=500)
        st.plotly_chart(fig_pie, use_container_width=True, key='three_disease_pie')

    # ----------------------------------------
    # 2-DISEASE ANALYSIS
    # ----------------------------------------
    elif analysis_mode == "two_diseases":
        st.markdown("---")
        
        # Create 2-disease combinations
        filtered_df['Disease_Combo_2'] = (
            filtered_df['has_diabetes'].astype(str) + '-' +
            filtered_df['has_cardio'].astype(str)
        )

        combo_labels_2 = {
            '0-0': 'No Disease',
            '1-0': 'Diabetes Only',
            '0-1': 'CVD Only',
            '1-1': 'Both Diabetes & CVD'
        }

        filtered_df['Disease_Label_2'] = filtered_df['Disease_Combo_2'].map(combo_labels_2)

        # Count combinations
        combo_counts_2 = filtered_df['Disease_Label_2'].value_counts().reset_index()
        combo_counts_2.columns = ['Disease Combination', 'Patient Count']
        combo_counts_2.insert(0, 'Rank', range(1, len(combo_counts_2) + 1))

        # ----------------------------------------
        # Summary Table
        # ----------------------------------------
        st.subheader("📋 Diabetes & CVD Comorbidity Summary")
        st.dataframe(combo_counts_2, use_container_width=True, hide_index=True, height=250)
        
        st.markdown("---")

        # ----------------------------------------
        # Pie Chart Only
        # ----------------------------------------
        st.subheader("🥧 Co-occurrence Breakdown")
        combo_with_disease_2 = combo_counts_2[combo_counts_2['Disease Combination'] != 'No Disease']
        
        fig_pie_2 = px.pie(
            combo_with_disease_2,
            values='Patient Count',
            names='Disease Combination',
            color_discrete_sequence=['#e74c3c', '#3498db', '#9b59b6']
        )
        fig_pie_2.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_2.update_layout(height=500)
        st.plotly_chart(fig_pie_2, use_container_width=True, key='two_disease_pie')
    
    # ----------------------------------------
    # INSUFFICIENT DATA
    # ----------------------------------------
    else:
        st.info("📊 Insufficient patient data for meaningful comorbidity analysis.")
        st.write(f"- Diabetes patients: {has_diabetes_count}")
        st.write(f"- CVD patients: {has_cvd_count}")
        st.write(f"- COPD patients: {has_copd_count}")
        st.write(f"\nMinimum {min_patients_threshold} patients per disease required for analysis.")
        # Add this as TAB 6 after the x_tab (Comorbidity) tab

# Update your existing tab assignments to use tabs_list[0], tabs_list[1], etc.

# ===============================
# TAB 6: INDIVIDUAL PATIENT CHECKING
# ===============================
with checking:  # Use the 6th tab
    st.header("🔍 Individual Patient Risk Checker")
    st.markdown("Search for a patient to view their comprehensive disease risk profile")
    st.markdown("---")

    
    # Patient Search
    col1, col2 = st.columns([3, 1])
    
    with col1:
        patient_id_search = st.text_input("🔍 Enter Patient ID", placeholder="e.g., a64e4a27-b68d-46f1-af01-1d4fe815a5b7")
    
    with col2:
        search_button = st.button("Search Patient", type="primary", use_container_width=True)
    
    # Search Logic
    if search_button or patient_id_search:
        # Search for patient
        patient_data = df[df['Id'].astype(str).str.contains(patient_id_search, case=False, na=False)]
        
        if len(patient_data) == 0:
            st.error(f"❌ No patient found with ID: {patient_id_search}")
            st.info("💡 Try entering a partial ID or check the Patient ID format")
            
        elif len(patient_data) > 1:
            st.warning(f"⚠️ Found {len(patient_data)} patients matching your search. Please be more specific.")
            st.dataframe(patient_data[['Id', 'AGE', 'GENDER', 'COUNTY']], use_container_width=True)
            
        else:
            # Exact match found
            patient = patient_data.iloc[0]
            
            st.success(f"✅ Patient Found: **{patient['Id']}**")
            st.markdown("---")
            
            # ========================================
            # DISEASE SUMMARY SECTION
            # ========================================
            st.subheader("🏥 Disease Status Summary")
            
            # Count how many diseases the patient has
            disease_count = int(patient['has_diabetes']) + int(patient['has_cardio']) + int(patient['has_copd'])
            
            # Disease status badges
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if disease_count == 0:
                    st.success("✅ Healthy\n\nNo Diseases")
                elif disease_count == 1:
                    st.warning(f"⚠️ 1 Disease\n\nSingle Condition")
                elif disease_count == 2:
                    st.error(f"🔴 2 Diseases\n\nComorbidity")
                else:
                    st.error(f"🚨 3 Diseases\n\nMultiple Comorbidity")
            
            with col2:
                if patient['has_diabetes'] == 1:
                    st.error("🩺 HAS\n\nDiabetes")
                else:
                    st.success("✅ NO\n\nDiabetes")
            
            with col3:
                if patient['has_cardio'] == 1:
                    st.error("🫀 HAS\n\nCVD")
                else:
                    st.success("✅ NO\n\nCVD")
            
            with col4:
                if patient['has_copd'] == 1:
                    st.error("🫁 HAS\n\nCOPD")
                else:
                    st.success("✅ NO\n\nCOPD")
            
            st.markdown("---")
            
            # ========================================
            # DEMOGRAPHICS
            # ========================================
            st.subheader("📋 Patient Demographics")
            
            col1, col2, col3, = st.columns(3)
            
            with col1:
                st.metric("Age", f"{patient['AGE']} years")
                st.metric("Gender", patient['GENDER'])
            
            with col2:
                st.metric("County", patient['COUNTY'])
                st.metric("State", patient['STATE'] if 'STATE' in patient else 'N/A')
            
            with col3:
                st.metric("BMI", f"{patient['BMI']:.1f}")
                bmi_category = patient['BMI_Category'] if 'BMI_Category' in patient else 'N/A'
                st.metric("BMI Category", bmi_category)
            
            st.markdown("---")
            
            # ========================================
            # DISEASE-SPECIFIC RISK ANALYSIS
            # ========================================
            st.subheader("📊 Detailed Risk Analysis by Disease")
            
            # Create 3 columns for each disease
            col1, col2, col3 = st.columns(3)
            
            # ========== DIABETES ==========
            with col1:
                st.markdown("### 🩺 Diabetes")
                
                # Risk Score Gauge
                diabetes_score = patient['diabetes_risk_score']
                diabetes_risk_bin = patient['Diabetes_Risk_Bin']
                
                # Color based on risk
                if diabetes_risk_bin == 'High Risk':
                    gauge_color = '#e74c3c'
                elif diabetes_risk_bin == 'Medium Risk':
                    gauge_color = '#f1c40f'
                else:
                    gauge_color = '#2ecc71'
                
                fig_diab = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=diabetes_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 16}},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "lightyellow"},
                            {'range': [66, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_diab.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_diab, use_container_width=True)
                
                # Risk Level Badge
                if diabetes_risk_bin == 'High Risk':
                    st.error(f"**Risk Level:** 🔴 {diabetes_risk_bin}")
                elif diabetes_risk_bin == 'Medium Risk':
                    st.warning(f"**Risk Level:** 🟡 {diabetes_risk_bin}")
                else:
                    st.success(f"**Risk Level:** 🟢 {diabetes_risk_bin}")
                
                # Key Metrics
                st.metric("Glucose", f"{patient['Glucose']:.0f} mg/dL")
                st.metric("HbA1c", f"{patient['Hemoglobin A1c/Hemoglobin.total in Blood']:.2f}%")
                
                # Status
                if patient['has_diabetes'] == 1:
                    st.error("**Status:** Diabetic")
                else:
                    st.success("**Status:** Non-Diabetic")
            
            # ========== CVD ==========
            with col2:
                st.markdown("### 🫀 Cardiovascular Disease")
                
                # Risk Score Gauge
                cvd_score = patient['Cardio_Risk_Score']
                cvd_risk_bin = patient['CVD_Risk_Bin']
                
                if cvd_risk_bin == 'High Risk':
                    gauge_color = '#e74c3c'
                elif cvd_risk_bin == 'Medium Risk':
                    gauge_color = '#f1c40f'
                else:
                    gauge_color = '#2ecc71'
                
                fig_cvd = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=cvd_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 16}},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 65], 'color': "lightyellow"},
                            {'range': [65, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig_cvd.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cvd, use_container_width=True)
                
                # Risk Level Badge
                if cvd_risk_bin == 'High Risk':
                    st.error(f"**Risk Level:** 🔴 {cvd_risk_bin}")
                elif cvd_risk_bin == 'Medium Risk':
                    st.warning(f"**Risk Level:** 🟡 {cvd_risk_bin}")
                else:
                    st.success(f"**Risk Level:** 🟢 {cvd_risk_bin}")
                
                # Key Metrics
                st.metric("Systolic BP", f"{patient['Systolic Blood Pressure']:.0f} mmHg")
                st.metric("Diastolic BP", f"{patient['Diastolic Blood Pressure']:.0f} mmHg")
                
                # Status
                if patient['has_cardio'] == 1:
                    st.error("**Status:** Has CVD")
                else:
                    st.success("**Status:** No CVD")
            
            # ========== COPD ==========
            with col3:
                st.markdown("### 🫁 COPD")
                
                # Risk Score Gauge
                copd_score = patient['COPD_Risk_Score']
                copd_risk_bin = patient['COPD_Risk_Bin']
                
                if copd_risk_bin == 'High Risk':
                    gauge_color = '#e74c3c'
                elif copd_risk_bin == 'Medium Risk':
                    gauge_color = '#f1c40f'
                else:
                    gauge_color = '#2ecc71'
                
                fig_copd = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=copd_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 16}},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 65], 'color': "lightyellow"},
                            {'range': [65, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig_copd.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_copd, use_container_width=True)
                
                # Risk Level Badge
                if copd_risk_bin == 'High Risk':
                    st.error(f"**Risk Level:** 🔴 {copd_risk_bin}")
                elif copd_risk_bin == 'Medium Risk':
                    st.warning(f"**Risk Level:** 🟡 {copd_risk_bin}")
                else:
                    st.success(f"**Risk Level:** 🟢 {copd_risk_bin}")
                
                # Key Metrics
                st.metric("Age", f"{patient['AGE']} years")
                
                # Risk Factors (if available)
                if 'risk_smoking_tobacco' in patient.index:
                    st.metric("Smoking Risk", f"{patient['risk_smoking_tobacco']:.2f}")
                if 'risk_respiratory_infection' in patient.index:
                    st.metric("Respiratory Risk", f"{patient['risk_respiratory_infection']:.2f}")
                
                # Status
                if patient['has_copd'] == 1:
                    st.error("**Status:** Has COPD")
                else:
                    st.success("**Status:** No COPD")
            
            st.markdown("---")
            
            # ========================================
            # RECOMMENDATIONS
            # ========================================
            st.subheader("💡 Health Recommendations")
            
            # Based on disease combination
            if disease_count == 0:
                st.success("""
                **✅ Excellent Health Status**
                - Maintain current healthy lifestyle
                - Continue regular health checkups
                - Keep monitoring key health metrics
                - Focus on preventive care
                """)
            
            elif disease_count == 1:
                if patient['has_diabetes'] == 1:
                    st.warning("""
                    **🩺 Diabetes Management**
                    - Monitor blood glucose regularly
                    - Follow prescribed medication schedule
                    - Maintain healthy diet (low sugar, high fiber)
                    - Regular exercise (30 min/day)
                    - HbA1c testing every 3 months
                    """)
                elif patient['has_cardio'] == 1:
                    st.warning("""
                    **🫀 Cardiovascular Care**
                    - Monitor blood pressure daily
                    - Reduce sodium intake
                    - Regular cardio exercise
                    - Stress management
                    - Medication compliance
                    """)
                else:
                    st.warning("""
                    **🫁 COPD Management**
                    - Avoid smoking and pollutants
                    - Use prescribed inhalers correctly
                    - Pulmonary rehabilitation exercises
                    - Get vaccinated (flu, pneumonia)
                    - Monitor oxygen levels
                    """)
            
            elif disease_count == 2:
                st.error("""
                **🔴 Multiple Disease Management Required**
                
                ⚠️ **URGENT: Comorbidity Detected**
                
                You have 2 co-existing conditions which require:
                - Integrated treatment plan from multiple specialists
                - More frequent monitoring (weekly checkups)
                - Strict medication adherence
                - Lifestyle modifications across all areas
                - Regular lab work and imaging
                
                **Schedule appointments with:**
                - Primary Care Physician
                - Disease-specific specialists
                - Nutritionist
                - Physical therapist
                """)
            
            else:  # 3 diseases
                st.error("""
                **🚨 CRITICAL: Triple Comorbidity**
                
                ⚠️ **IMMEDIATE MEDICAL ATTENTION REQUIRED**
                
                You have all 3 conditions which creates complex health risks:
                - High priority for coordinated care team
                - Daily monitoring required
                - Strict treatment protocol adherence
                - Possible hospitalization risk
                - Immediate lifestyle intervention needed
                
                **URGENT ACTIONS:**
                1. Schedule emergency consultation with primary care
                2. Enroll in intensive disease management program
                3. Daily vital signs monitoring
                4. Strict medication schedule
                5. Consider home health aide
                6. Emergency contact list prepared
                """)
            
            st.markdown("---")