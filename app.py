import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="AI/ML Salary Predictor", layout="wide")

# --- Data Loading and Preprocessing with Caching ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("adult3.csv")
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

@st.cache_resource
def encode_and_train_models(df):
    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    results = []
    model_dict = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
        model_dict[name] = model
    results_df = pd.DataFrame(results)
    return label_encoders, X, model_dict, results_df, df_encoded

# --- Introduction ---
st.title("AI/ML Employee Salary Prediction App")
st.markdown("""
This application predicts if a person's salary is above $50,000/year, featuring robust visualizations, feature explainers, and improved usability.
""")

# --- Data Dictionary Modal ---
with st.expander("ℹ️ Data Dictionary (Click to expand)", expanded=False):
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

# --- File Upload or Default Dataset ---
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
df = load_and_clean_data(uploaded_file)
if uploaded_file:
    st.info("Using uploaded file.")
else:
    st.info("Using default dataset (adult3.csv).")

# --- Data Preview ---
st.subheader("Preview of Data")
st.dataframe(df.head())

# --- Encode and Train Models (Cached) ---
label_encoders, X, model_dict, results_df, df_encoded = encode_and_train_models(df)

def validate_user_input(age, hours):
    errors = []
    if age < 18 or age > 70:
        errors.append("Age must be between 18 and 70.")
    if hours < 1 or hours > 99:
        errors.append("Hours-per-week must be between 1 and 99.")
    return errors

# --- Model Performance Plots (Interactive) ---
st.subheader("Model Performance Comparison")
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

# --- Graph Display Toggles ---
st.subheader("Explore Additional Visualizations")
show_age = st.checkbox("Show Income by Age (Boxplot)", value=True)
show_country = st.checkbox("Show Avg. Income by Country", value=False)
show_occupation = st.checkbox("Show Top Paying Occupations", value=False)
show_dist = st.checkbox("Show Income Class Distribution", value=True)

# --- Age vs Income Interactive Boxplot ---
if show_age and 'age' in df_encoded.columns:
    df_age = df_encoded.copy()
    df_age['income'] = label_encoders['income'].inverse_transform(df_encoded['income'])
    fig_age = px.box(df_age, x='income', y='age', points="all",
                     color='income',
                     labels={"income": "Income Class", "age": "Age"},
                     title="Income vs Age")
    st.plotly_chart(fig_age, use_container_width=True)
    st.caption("Younger groups are mostly below $50K. Hover for details.")

# --- Country vs Income ---
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

# --- Occupation vs Income ---
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

# --- Income Class Distribution ---
if show_dist:
    income_counts = df_encoded['income'].value_counts(normalize=True) * 100
    income_labels = label_encoders['income'].inverse_transform(income_counts.index)
    fig_income = px.bar(x=income_labels, y=income_counts.values, text=income_counts.round(1),
                        labels={"x": "Income Class", "y": "Percentage"},
                        title="Proportion of Income Classes")
    st.plotly_chart(fig_income, use_container_width=True)
    st.caption("Shows class balance. Hover for precise percentages.")


# --- User Input Form, With Validation ---
st.subheader("Try It Yourself: Salary Prediction")
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
        # --- Input Validation ---
        errors = validate_user_input(age, hours)
        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                input_data = {
                    'age': age,
                    'education': label_encoders['education'].transform([education])[0],
                    'occupation': label_encoders['occupation'].transform([occupation])[0],
                    'relationship': label_encoders['relationship'].transform([relationship])[0],
                    'hours-per-week': hours,
                    'native-country': label_encoders['native-country'].transform([country])[0],
                    'workclass': label_encoders['workclass'].transform([workclass])[0],
                    'gender': label_encoders['gender'].transform([gender])[0],
                }
                for col in X.columns:
                    if col not in input_data:
                        input_data[col] = X[col].mean()

                input_df = pd.DataFrame([input_data])[X.columns]
                model = model_dict[model_choice]
                prediction = model.predict(input_df)[0]
                label = label_encoders['income'].inverse_transform([prediction])[0]
                st.success(f"{name}, your predicted income range is: {label}")
                st.info(f"You used the {model_choice} model.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- Footer ---
st.markdown("""
<hr style="border: 0.5px solid gray; margin-top: 30px;" />
<div style="text-align: center; padding: 10px; font-size: 14px; color: #888;">
    Salary Prediction Using AI/ML &nbsp; &#169;
</div>
""", unsafe_allow_html=True)
