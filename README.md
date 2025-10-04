# AI/ML Employee Salary Prediction App

This repository contains an interactive, explainable web application that predicts whether an individual's salary exceeds **$50,000/year** using demographic and job-related data.  
Built with **Python** and **Streamlit**, the app offers a modern, sidebar-driven interface with rich visualizations, model comparisons, and the ability to upload your own datasets.

---

## Features

- **Intuitive Sidebar Navigation**
  - Easily explore sections for:
    - Model Performance
    - Visualizations
    - Job Listings
    - Salary Predictor
- **Upload Your Own Dataset** (`.csv`) or use the built-in **Adult Income dataset**
- **Live Predictions** with custom user input and model selection (Logistic Regression, Random Forest, Gradient Boosting)
- **Interactive Visualizations** powered by Plotly for:
  - Accuracy, Precision, Recall, and F1 Score comparison
  - Income analysis by age, country, occupation
  - Income distribution and class imbalance insights
- **Explainable AI (XAI):** SHAP-based feature impact explanations for transparent predictions
- **Optimized Performance** with caching for data and model computations
- **Dark/Light Theme Support** with responsive design for all screen sizes

---

## Project Structure

| File/Folder        | Description                |
| ------------------ | -------------------------- |
| `app.py`           | Main Streamlit application |
| `requirements.txt` | Python dependencies        |
| `adult3.csv`       | Default dataset            |
| `README.md`        | Project documentation      |

---

## Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/Amogh9000/salarypred.git
    cd salarypred
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

4. **Open in your browser:**  
   [http://localhost:8501](http://localhost:8501) (default)

---

## Usage Guide

**Navigate Using the Sidebar:**  
Choose from the following sections:

- **Model Performance:** Compare accuracy, precision, recall, and F1 scores across models.  
- **Visualizations:** Explore interactive charts and data distributions.  
- **Job Listings:** Discover sample job data insights related to income trends.  
- **Salary Predictor:** Enter your own demographic and job attributes to predict income.  

**Upload Your Own Data:**  
Bring in your own dataset (with similar columns) for personalized analysis.  

**Customize Models:**  
Switch between Random Forest, Logistic Regression, or Gradient Boosting to evaluate and compare model performance.  

---

## Troubleshooting & Tips

- **App Feels Slow?**  
  Large files or multiple model evaluations can cause temporary lag — try smaller samples or fewer visuals.

- **Missing Packages?**  
  Run `pip install -r requirements.txt` again to fix missing dependencies (e.g., `plotly`, `shap`, etc.).

---

## Customization

- Add or modify **Plotly visuals** in `app.py`
- Tune **model parameters** for better accuracy or interpretability
- Expand the **Job Listings** section with real APIs or datasets
- **Deploy publicly** using Streamlit Community Cloud, Docker, or cloud platforms like AWS or Vercel

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Dataset:** UCI / Kaggle – Adult Income Dataset  
- **Libraries:** Streamlit, scikit-learn, Plotly, pandas, imbalanced-learn, SHAP  
- **Community:** Open-source ML & data science contributors  

---

**Project Date:** July 2025  
**Updated:** October 2025 (Sidebar & Job Listings Integration)
