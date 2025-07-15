import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- Page Configuration ---
st.set_page_config(page_title="AI/ML Salary Predictor", layout="wide")

# --- Custom Dark Theme ---
st.markdown("""
<style>
body {
    margin: 0;
    overflow: hidden;
}
.stApp {
    background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
    height: 100vh;
    overflow: hidden;
    color: white;
    position: relative;
}
.stars {
    width: 1.1px;
    height: 1.7px;
    background: transparent;
    box-shadow: 
    100px 200px white, 150px 300px white, 200px 400px white, 250px 500px white,
    300px 600px white, 350px 700px white, 400px 800px white, 450px 900px white,
    500px 1000px white, 550px 1100px white, 600px 1200px white, 650px 1300px white,
    700px 1400px white, 750px 1500px white, 800px 1600px white, 850px 1700px white,
    900px 1800px white, 950px 1900px white, 1000px 2000px white, 1050px 2100px white;
    animation: animateStars 50s linear infinite;
    position: absolute;
}
@keyframes animateStars {
    from {transform: translateY(0);}
    to {transform: translateY(-2000px);}
}
.main .block-container {
    max-width: 1100px;
    margin: auto;
}
</style>
<div class="stars"></div>
""", unsafe_allow_html=True)

# --- Introduction ---
st.title("AI/ML Employee Salary Prediction App")
st.markdown("""
This application uses machine learning models to predict whether a person's salary is likely to be more than $50,000/year 
based on their demographic and employment-related features. It includes performance comparisons and feature importance visualizations.
""")

def center_plot(fig, max_width="1000px"):
    st.markdown(
        f"""<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
            <div style="max-width: {max_width}; width: 100%;">""",
        unsafe_allow_html=True,
    )
    st.pyplot(fig)
    st.markdown("</div></div>", unsafe_allow_html=True)

# --- File Upload or Default Dataset ---
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.info("Using uploaded file.")
else:
    df = pd.read_csv("adult3.csv")
    st.info("No file uploaded. Using default dataset (adult3.csv).")

# --- Preprocessing ---
st.subheader("Preview of Data")
st.dataframe(df.head())

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("income", axis=1)
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})

results_df = pd.DataFrame(results)

# --- Model Accuracy Barplot ---
st.subheader("Model Performance Comparison")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x="Model", y="Accuracy", data=results_df, ax=ax1, palette="plasma")
    ax1.set_ylim(0, 1)
    ax1.set_title("Accuracy by Model")
    st.pyplot(fig1)
    st.markdown(
        "This bar graph compares the prediction accuracy of different machine learning models. "
        "A higher accuracy indicates better overall performance in predicting whether an individual earns above or below $50,000."
    )

with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x="Model", y="F1 Score", data=results_df, ax=ax2, palette="crest")
    ax2.set_ylim(0, 1)
    ax2.set_title("F1 Score by Model")
    st.pyplot(fig2)
    st.markdown(
        "F1 Score considers both precision and recall, making it useful for evaluating imbalanced datasets. "
        "This chart highlights which model best balances false positives and false negatives."
    )


# --- Age vs Income ---
if 'age' in df.columns:
    st.subheader("Income by Age")
    df_age = df.copy()
    df_age['income'] = label_encoders['income'].inverse_transform(df['income'])
    fig_age, ax_age = plt.subplots(figsize=(5.5, 3.5))
    sns.boxplot(x='income', y='age', data=df_age, ax=ax_age, palette="Set2")
    ax_age.set_title("Income vs Age")
    st.markdown(
    "This boxplot shows the age distribution for individuals earning above and below $50,000. "
    "Typically, higher earners fall in older age brackets, suggesting experience may play a role in income."
)
    center_plot(fig_age)

# --- Country vs Income ---
if 'native-country' in df.columns:
    st.subheader("Average Income by Country")
    df_country = df.copy()
    df_country['native-country'] = label_encoders['native-country'].inverse_transform(df['native-country'])
    country_income = df_country.groupby('native-country')['income'].mean().sort_values(ascending=False)
    fig_country, ax_country = plt.subplots(figsize=(8.5, 3.5))
    country_income.plot(kind='bar', ax=ax_country, color='teal')
    ax_country.set_ylabel("Avg. Income (Encoded)")
    ax_country.set_title("Average Income by Country")
    st.markdown(
    "This bar chart displays the average encoded income class for each country in the dataset. "
    "Higher values suggest a greater proportion of individuals earning above $50,000 from that country."
)
    center_plot(fig_country)

# --- Occupation vs Income ---
if 'occupation' in df.columns:
    st.subheader("Top Paying Occupations")
    df_occ = df.copy()
    df_occ['occupation'] = label_encoders['occupation'].inverse_transform(df['occupation'])
    occ_income = df_occ.groupby('occupation')['income'].mean().sort_values(ascending=False)
    fig_occ, ax_occ = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(x=occ_income.values, y=occ_income.index, ax=ax_occ, palette="coolwarm")
    ax_occ.set_title("Avg. Income by Occupation")
    st.markdown("This horizontal bar graph highlights the average income levels by occupation. It reveals which job roles are associated with higher income classes, with roles like 'Exec-managerial' or 'Prof-specialty' often ranking higher.")
    center_plot(fig_occ)

# --- Feature Importance ---
st.subheader("Feature Importance (Random Forest)")
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
fig_imp, ax_imp = plt.subplots(figsize=(6.5, 4.5))
importances.plot(kind='barh', ax=ax_imp, color="skyblue")
ax_imp.set_title("Feature Importance")
st.markdown(
    "This horizontal bar graph highlights the average income levels by occupation. "
    "It reveals which job roles are associated with higher income classes, with roles like 'Exec-managerial' or 'Prof-specialty' often ranking higher."
)
center_plot(fig_imp)

# --- Income Class Distribution ---
st.subheader("Income Class Distribution")
income_counts = df['income'].value_counts(normalize=True) * 100
income_labels = label_encoders['income'].inverse_transform(income_counts.index)
fig_income, ax_income = plt.subplots(figsize=(6, 2.5))
sns.barplot(x=income_counts.values, y=income_labels, palette='viridis', ax=ax_income)
for i, v in enumerate(income_counts.values):
    ax_income.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
ax_income.set_xlabel("Percentage")
ax_income.set_xlim(0, 100)
ax_income.set_title("Proportion of Income Classes")
st.markdown("""
This bar chart shows the distribution of income classes in the dataset.
- The '>50K' class represents individuals earning above $50,000.
- The '<=50K' class represents individuals earning $50,000 or less.
- The chart shows the percentage of each income class in the dataset.
""")
center_plot(fig_income)

# --- Correlation Heatmap ---
st.subheader("Feature Correlation Heatmap")
st.markdown("""
**What This Shows:**

This heatmap displays the correlation between different numerical features in the dataset.

- A correlation value ranges from **-1 to +1**.
- A value **close to +1** means a **strong positive relationship** — as one feature increases, so does the other.
- A value **close to -1** indicates a **strong negative relationship** — as one feature increases, the other decreases.
- Values near **0** suggest **little to no linear relationship** between the two features.

**How This Helps:**

By examining this matrix, we can:
- Identify features that may be redundant (highly correlated with each other).
- Understand relationships, such as whether working more hours per week correlates with higher income.
- Make informed decisions about feature selection or dimensionality reduction.
""")
fig_corr, ax_corr = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax_corr)
center_plot(fig_corr)

# --- User Input Form ---
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
    submit = st.form_submit_button("Predict My Income")

    if submit:
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
            prediction = rf_model.predict(input_df)[0]
            label = label_encoders['income'].inverse_transform([prediction])[0]
            st.success(f"{name}, your predicted income range is: {label}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.success("Done! You can explore more or upload another dataset.")
# --- Footer ---
st.markdown("""
<hr style="border: 0.5px solid gray; margin-top: 30px;" />
<div style="text-align: center; padding: 10px; font-size: 14px; color: #888;">
    Salary Prediction Using AI/ML &nbsp; &#169;
</div>
""", unsafe_allow_html=True)

