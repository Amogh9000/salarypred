import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px
from imblearn.over_sampling import SMOTE

# --- Page Configuration ---
st.set_page_config(page_title="AI/ML Salary Predictor", layout="wide")

# --- Data Loading and Preprocessing with Caching ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Load CSV (uploaded or default). Keep same cleaning you used:
    replace '?' with NaN and drop rows with NaN.
    """
    if uploaded_file:
        df_local = pd.read_csv(uploaded_file)
    else:
        df_local = pd.read_csv("adult3.csv")
    df_local.replace("?", np.nan, inplace=True)
    df_local.dropna(inplace=True)
    return df_local

@st.cache_resource
def encode_and_train_models(df):
    """
    EXACTLY your original encoding / SMOTE / training pipeline, cached.
    Returns label_encoders, X (features), model_dict, results_df, df_encoded.
    """
    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Keep your original target column name as in your working code:
    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]

    # Split first, then SMOTE on training only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    results = []
    model_dict = {}
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
        model_dict[name] = model

    results_df = pd.DataFrame(results)
    return label_encoders, X, model_dict, results_df, df_encoded

# --- Load data ---
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
df = load_and_clean_data(uploaded_file)
if uploaded_file:
    st.info("Using uploaded file.")
else:
    st.info("Using default dataset (adult3.csv).")

# --- Encode & Train (cached) ---
label_encoders, X, model_dict, results_df, df_encoded = encode_and_train_models(df)

# --- Input validation helper (unchanged) ---
def validate_user_input(age, hours):
    errors = []
    if age < 18 or age > 70:
        errors.append("Age must be between 18 and 70.")
    if hours < 1 or hours > 99:
        errors.append("Hours-per-week must be between 1 and 99.")
    return errors

# --- Sidebar Navigation (only addition) ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "Dataset Dictionary",
    "Dataset Overview",
    "Model Performance",
    "Visualizations",
    "Salary Predictor",
    "Job Listings"
])
# ---------------------------
# Pages (preserve original behavior)
# ---------------------------

if page == "Dataset Dictionary":
    st.header("Data Dictionary")
    st.markdown("""
    | Feature Name       | Description                                               |
    |--------------------|-----------------------------------------------------------|
    | age                | Age of the individual                                    |
    | workclass          | Type of employer (e.g., Private, Self-emp, Govt)         |
    | education          | Highest educational attainment                           |
    | occupation         | Occupation (job role)                                    |
    | relationship       | Relationship or family status                            |
    | hours-per-week     | Work hours per week                                      |
    | native-country     | Country of origin                                        |
    | gender             | Gender (Male/Female/Other)                               |
    | income             | Target: Salary (<=50K or >50K)                           |
    """)

elif page == "Dataset Overview":
    st.header("Preview of Data")
    st.dataframe(df.head(), use_container_width=True)
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

elif page == "Model Performance":
    st.header("Model Performance Comparison")
    # show the results table and the two bar charts you had earlier
    st.dataframe(results_df.set_index("Model"))
    tab1, tab2 = st.tabs(["Accuracy", "F1 Score"])
    with tab1:
        fig1 = px.bar(results_df, x="Model", y="Accuracy", color="Model", text="Accuracy",
                      labels={"Accuracy": "Accuracy Score"}, title="Accuracy by Model")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Hover for exact accuracy. Higher is better.")
    with tab2:
        fig2 = px.bar(results_df, x="Model", y="F1 Score", color="Model", text="F1 Score",
                      labels={"F1 Score": "F1 Score"}, title="F1 Score by Model")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Hover for exact F1 score. Higher balances precision and recall.")

elif page == "Visualizations":
    st.header("Explore Additional Visualizations")
    show_age = st.checkbox("Show Income by Age (Boxplot)", value=True)
    show_country = st.checkbox("Show Avg. Income by Country", value=False)
    show_occupation = st.checkbox("Show Top Paying Occupations", value=False)
    show_dist = st.checkbox("Show Income Class Distribution", value=True)

    # Use your encoded df + the label_encoders to inverse transform exactly as before
    if show_age and 'age' in df_encoded.columns:
        df_age = df_encoded.copy()
        df_age['income'] = label_encoders['income'].inverse_transform(df_encoded['income'])
        fig_age = px.box(df_age, x='income', y='age', points="all",
                         color='income',
                         labels={"income": "Income Class", "age": "Age"},
                         title="Income vs Age")
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption("Younger groups are mostly below $50K. Hover for details.")

    if show_country and 'native-country' in df_encoded.columns:
        df_country = df_encoded.copy()
        df_country['native-country'] = label_encoders['native-country'].inverse_transform(df_encoded['native-country'])
        country_income = df_country.groupby('native-country')['income'].mean().sort_values(ascending=False)
        fig_country = px.bar(country_income,
                            orientation="v",
                            labels={"value": "Avg. Encoded Income", "index": "Country"},
                            title="Average Income by Country")
        st.plotly_chart(fig_country, use_container_width=True)
        st.caption("Higher values mean a greater proportion of higher earners from that country.")

    if show_occupation and 'occupation' in df_encoded.columns:
        df_occ = df_encoded.copy()
        df_occ['occupation'] = label_encoders['occupation'].inverse_transform(df_encoded['occupation'])
        occ_income = df_occ.groupby('occupation')['income'].mean().sort_values(ascending=False)
        fig_occ = px.bar(occ_income,
                        orientation="h",
                        labels={"value": "Avg. Encoded Income", "index": "Occupation"},
                        title="Average Income by Occupation")
        st.plotly_chart(fig_occ, use_container_width=True)
        st.caption("Top jobs by income. Hover for details.")

    if show_dist:
        income_counts = df_encoded['income'].value_counts(normalize=True) * 100
        income_labels = label_encoders['income'].inverse_transform(income_counts.index)
        fig_income = px.bar(x=income_labels, y=income_counts.values, text=income_counts.round(1),
                            labels={"x": "Income Class", "y": "Percentage"},
                            title="Proportion of Income Classes")
        st.plotly_chart(fig_income, use_container_width=True)
        st.caption("Shows class balance. Hover for precise percentages.")

elif page == "Salary Predictor":
    st.header("Salary Prediction")
    st.write("Enter your details and select a model to predict income:")

    # Keep EXACT UI fields and encoding logic you originally used:
    with st.form("prediction_form"):
        name = st.text_input("Your Name")
        age = st.slider("Age", 18, 70, 30)
        education = st.selectbox("Education", label_encoders['education'].classes_)
        occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
        relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)
        hours = st.slider("Hours/Week", 1, 99, 40)
        country = st.selectbox("Country", label_encoders['native-country'].classes_)
        workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
        gender = st.selectbox("Gender", label_encoders['gender'].classes_)
        model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression", "Gradient Boosting"])
        submit = st.form_submit_button("Predict My Income")

        if submit:
            errors = validate_user_input(age, hours)
            if errors:
                for err in errors:
                    st.error(err)
            else:
                try:
                    # Build input using the same column keys as your training X
                    input_data = {}
                    input_data['age'] = age
                    input_data['education'] = label_encoders['education'].transform([education])[0]
                    input_data['occupation'] = label_encoders['occupation'].transform([occupation])[0]
                    input_data['relationship'] = label_encoders['relationship'].transform([relationship])[0]
                    input_data['hours-per-week'] = hours
                    input_data['native-country'] = label_encoders['native-country'].transform([country])[0]
                    input_data['workclass'] = label_encoders['workclass'].transform([workclass])[0]
                    input_data['gender'] = label_encoders['gender'].transform([gender])[0]

                    # Fill missing columns with means exactly as before
                    for col in X.columns:
                        if col not in input_data:
                            input_data[col] = X[col].mean()

                    input_df = pd.DataFrame([input_data])[X.columns]

                    # Prediction using cached models
                    model = model_dict[model_choice]
                    prediction = model.predict(input_df)[0]
                    label = label_encoders['income'].inverse_transform([prediction])[0]

                    # Probability-based salary approximation (unchanged)
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_df)[0]
                        low_anchor_usd = 35000.0
                        high_anchor_usd = 80000.0
                        hours_factor = max(0.0, hours / 40.0)

                        unpaid_labels = ["Without-pay", "Never-worked"]
                        try:
                            is_unpaid = any([lbl in label_encoders['workclass'].classes_ and
                                             label_encoders['workclass'].transform([lbl])[0] == input_data['workclass']
                                             for lbl in unpaid_labels])
                        except Exception:
                            is_unpaid = False

                        if is_unpaid:
                            expected_salary_annual = 0.0
                            st.warning("Workclass indicates unpaid/never-worked; expected salary set to $0.")
                        else:
                            expected_salary_annual = (proba[1] * high_anchor_usd + proba[0] * low_anchor_usd) * hours_factor

                        if hours < 10:
                            st.warning("Very low hours/week detected; expected salary scaled down proportionally to hours.")
                        if hours > 80:
                            st.warning("Unusually high hours/week detected; result is extrapolated from a 40h baseline.")

                        # Left column: <=50K; Right column: >50K (same ranges you used)
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("Income ≤ $50K")
                            st.markdown(f"Probability: {proba[0]:.1%}")
                            if proba[0] >= 0.9:
                                estimated_salary = "$20,000 - $30,000"; confidence = "Very High"
                            elif proba[0] >= 0.8:
                                estimated_salary = "$25,000 - $35,000"; confidence = "High"
                            elif proba[0] >= 0.7:
                                estimated_salary = "$30,000 - $40,000"; confidence = "Medium-High"
                            elif proba[0] >= 0.6:
                                estimated_salary = "$35,000 - $45,000"; confidence = "Medium"
                            else:
                                estimated_salary = "$40,000 - $50,000"; confidence = "Low"
                            st.metric("Estimated Salary", estimated_salary, f"Confidence: {confidence}")
                            st.info(f"Based on {proba[0]:.1%} probability of earning ≤$50K")

                        with col2:
                            st.markdown("Income > $50K")
                            st.markdown(f"Probability: {proba[1]:.1%}")
                            if proba[1] >= 0.9:
                                estimated_salary = "$80,000 - $150,000+"; confidence = "Very High"
                            elif proba[1] >= 0.8:
                                estimated_salary = "$70,000 - $120,000"; confidence = "High"
                            elif proba[1] >= 0.7:
                                estimated_salary = "$60,000 - $100,000"; confidence = "Medium-High"
                            elif proba[1] >= 0.6:
                                estimated_salary = "$55,000 - $85,000"; confidence = "Medium"
                            else:
                                estimated_salary = "$50,000 - $75,000"; confidence = "Low"
                            st.metric("Estimated Salary", estimated_salary, f"Confidence: {confidence}")
                            st.info(f"Based on {proba[1]:.1%} probability of earning >$50K")

                        # Final summary (same as original)
                        st.markdown("---")
                        if proba[0] > proba[1]:
                            most_likely_range = "≤ $50,000"
                            if proba[0] >= 0.9:
                                most_likely_salary = "$20,000 - $30,000"; confidence_level = "Very High"
                            elif proba[0] >= 0.8:
                                most_likely_salary = "$25,000 - $35,000"; confidence_level = "High"
                            elif proba[0] >= 0.7:
                                most_likely_salary = "$30,000 - $40,000"; confidence_level = "Medium-High"
                            elif proba[0] >= 0.6:
                                most_likely_salary = "$35,000 - $45,000"; confidence_level = "Medium"
                            else:
                                most_likely_salary = "$40,000 - $50,000"; confidence_level = "Low"
                        else:
                            most_likely_range = "> $50,000"
                            if proba[1] >= 0.9:
                                most_likely_salary = "$80,000 - $150,000+"; confidence_level = "Very High"
                            elif proba[1] >= 0.8:
                                most_likely_salary = "$70,000 - $120,000"; confidence_level = "High"
                            elif proba[1] >= 0.7:
                                most_likely_salary = "$60,000 - $100,000"; confidence_level = "Medium-High"
                            elif proba[1] >= 0.6:
                                most_likely_salary = "$55,000 - $85,000"; confidence_level = "Medium"
                            else:
                                most_likely_salary = "$50,000 - $75,000"; confidence_level = "Low"

                        st.markdown("### Final Prediction Summary")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Predicted Class", label, f"Probability: {max(proba):.1%}")
                        c2.metric("Most Likely Salary", most_likely_salary, f"Confidence: {confidence_level}")
                        c3.metric("Model Used", model_choice, "AI-Powered")
                        c4.metric("Expected Salary (annual, USD)", f"${expected_salary_annual:,.0f}")

                        st.success(f"{name}, your predicted income range is: {label}")
                        st.info(f"Most Likely Salary: {most_likely_salary} | Confidence: {confidence_level} | Model: {model_choice}")
                    else:
                        st.success(f"{name}, your predicted income range is: {label}")
                        st.info(f"You used the {model_choice} model.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

elif page == "Job Listings":
    st.header("Job Search & Career Opportunities")
    job_keywords = st.text_input("Enter job keywords (e.g., 'machine learning engineer', 'data scientist', 'AI developer')",
                                placeholder="machine learning engineer")
    country_option = st.selectbox("Select country for job listings", ["Global", "India", "United States", "United Kingdom", "Canada", "Australia", "Germany"])

    if job_keywords:
        st.markdown("Available Job Platforms")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            ["LinkedIn", "Indeed", "Glassdoor", "Tech Jobs", "Remote Jobs", "Freelance", "International", "All Platforms"])

        kw_q = job_keywords.replace(' ', '+')
        kw_pct = job_keywords.replace(' ', '%20')
        def indeed_domain(country):
            return {
                "India": f"https://in.indeed.com/jobs?q={kw_q}",
                "United States": f"https://www.indeed.com/jobs?q={kw_q}",
                "United Kingdom": f"https://uk.indeed.com/jobs?q={kw_q}",
                "Canada": f"https://ca.indeed.com/jobs?q={kw_q}",
                "Australia": f"https://au.indeed.com/jobs?q={kw_q}",
                "Germany": f"https://de.indeed.com/jobs?q={kw_q}",
            }.get(country, f"https://www.indeed.com/jobs?q={kw_q}")

        def glassdoor_domain(country):
            return {
                "India": f"https://www.glassdoor.co.in/Job/jobs.htm?sc.keyword={kw_pct}",
                "United States": f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={kw_pct}",
                "United Kingdom": f"https://www.glassdoor.co.uk/Job/jobs.htm?sc.keyword={kw_pct}",
                "Canada": f"https://www.glassdoor.ca/Job/jobs.htm?sc.keyword={kw_pct}",
                "Australia": f"https://www.glassdoor.com.au/Job/jobs.htm?sc.keyword={kw_pct}",
                "Germany": f"https://www.glassdoor.de/Job/jobs.htm?sc.keyword={kw_pct}",
            }.get(country, f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={kw_pct}")

        def linkedin_url(country):
            loc = country if country != "Global" else ""
            loc_param = f"&location={loc.replace(' ', '%20')}" if loc else ""
            return f"https://www.linkedin.com/jobs/search/?keywords={kw_pct}{loc_param}"

        def angel_url(country):
            loc = country if country != "Global" else ""
            loc_part = f"&locations%5B%5D={loc.replace(' ', '%20')}" if loc else ""
            return f"https://wellfound.com/jobs?search={kw_q}{loc_part}"

        with tab1:
            st.markdown("LinkedIn Jobs")
            st.markdown(f"- [Search LinkedIn Jobs]({linkedin_url(country_option)})")
            st.markdown(f"- [People Search]({'https://www.linkedin.com/search/results/people/?keywords=' + kw_pct})")
            st.info("LinkedIn offers jobs, professional networking, and industry insights.")

        with tab2:
            st.markdown("Indeed Jobs")
            st.markdown(f"- [Indeed Jobs]({indeed_domain(country_option)})")
            st.markdown(f"- [Salary Information]({'https://www.indeed.com/career/' + job_keywords.replace(' ', '-') + '/salaries'})")
            st.info("Indeed provides comprehensive job listings and salary data.")

        with tab3:
            st.markdown("Glassdoor")
            st.markdown(f"- [Glassdoor Jobs]({glassdoor_domain(country_option)})")
            st.info("Glassdoor offers jobs, salary insights, and company reviews.")

        with tab4:
            st.markdown("Tech-Specific Jobs")
            st.markdown(f"- [Wellfound (AngelList)]({angel_url(country_option)})")
            st.markdown(f"- [Dice]({'https://www.dice.com/jobs?q=' + kw_q})")
            st.info("Tech-specific platforms for software, AI, and engineering roles.")

        with tab5:
            st.markdown("Remote Work Opportunities")
            st.markdown(f"- [Remote.co]({'https://remote.co/remote-jobs/search/?search_keywords=' + kw_q})")
            st.markdown(f"- [We Work Remotely]({'https://weworkremotely.com/remote-jobs/search?term=' + kw_q})")
            st.info("Platforms specializing in remote and work-from-home opportunities.")

        with tab6:
            st.markdown("Freelance & Contract Work")
            st.markdown(f"- [Upwork]({'https://www.upwork.com/search/jobs/?q=' + kw_q})")
            st.markdown(f"- [Fiverr]({'https://www.fiverr.com/search/gigs?query=' + kw_q})")
            st.info("Freelance and contract work opportunities for independent professionals.")

        with tab7:
            st.markdown("International Job Markets")
            st.markdown(f"- [Indeed ({country_option})]({indeed_domain(country_option)})")
            st.markdown(f"- [LinkedIn ({country_option})]({linkedin_url(country_option)})")
            st.markdown(f"- [Glassdoor ({country_option})]({glassdoor_domain(country_option)})")
            st.info("Job platforms tailored for the selected country when available.")

        with tab8:
            st.markdown("All Platforms")
            st.markdown(f"- [LinkedIn]({linkedin_url(country_option)})")
            st.markdown(f"- [Indeed]({indeed_domain(country_option)})")
            st.markdown(f"- [Glassdoor]({glassdoor_domain(country_option)})")
            st.markdown(f"- [Wellfound]({angel_url(country_option)})")
            st.success("Click any link to open the job search in a new tab!")

# --- Footer ---
st.markdown("""
<hr style="border: 0.5px solid gray; margin-top: 30px;" />
<div style="text-align: center; padding: 10px; font-size: 14px; color: #888;">
    Salary Prediction Using AI/ML &nbsp; ©
</div>
""", unsafe_allow_html=True)
