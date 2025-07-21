# AI/ML Employee Salary Prediction App

This repository contains a robust, interactive web application that predicts whether an individual's salary exceeds $50,000/year using demographic and job-related data. The project is built with Python and Streamlit, offering a modern, user-friendly interface with rich visualizations and the ability to upload your own datasets.

---

## 🚀 Features

- **User-friendly interface** with modern dark theme and responsive design
- **Upload your dataset** (`.csv`) or use the built-in sample (Adult Income dataset)
- **Live predictions** with custom user input and choice of machine learning model (Logistic Regression, Random Forest, Gradient Boosting)
- **Interactive visualizations** powered by Plotly for:
  - Model accuracy and F1 score comparison
  - Income analysis by age, country, occupation
  - Feature importance and correlation heatmap
  - Class distribution in the dataset
- **Data dictionary** and clear data previews for user guidance
- **Optimized performance** with smart data/model caching using Streamlit

---

###  🏗️ Project Structure

project-root/
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
├── adult3.csv         # Default dataset
└── README.md          # Project documentation





---

## 📦 Installation

1. **Clone this repository:**
    ```
    git clone https://github.com/Amogh9000/salarypred.git
    cd salarypred
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```
    streamlit run app.py
    ```

4. **Open in your browser:**  
   [localhost:8501](http://localhost:8501) (by default)

---

## 📝 Usage

- **Preview Default Data:** View summary, stats, and structure of the built-in dataset.
- **Upload Your Own Data:** Use the upload button to bring in your own CSV with similar columns.
- **Visualize Insights:** Explore interactive charts explaining model performance and data relationships.
- **Try Live Prediction:** Enter your own demographic and employment attributes in the "Try it yourself" section and see model predictions in real time.
- **Select Model:** Choose between Random Forest, Logistic Regression, or Gradient Boosting for your prediction.

---

# ⚠️ Troubleshooting & Tips

- **Laggy Performance:** Visualizations over large files or training multiple models can make the app temporarily slow. Use streamlined data and adjust checkboxes to view only the necessary charts.
- **Module Not Found:** If you get errors like `ModuleNotFoundError: No module named 'plotly'`, ensure you installed all packages from `requirements.txt`.
- **Class Imbalance Note:** The built-in dataset is imbalanced. Most predictions may fall into the "`<=50K`" category unless using class balancing techniques.

---

## 🛠️ Customization

- **Add Your Own Visuals** by expanding the Plotly plots in `app.py`.
- **Adjust Model Parameters** for different performance or fairness criteria.
- **Deploy Publicly:** You can use Streamlit Community Cloud, Vercel (with a Python backend), or similar for free hosting and sharing.

---

## 📚 License

This project is open-source.

---

## 🤝 Acknowledgements

- UCI Machine Learning Repository – Adult Income Dataset
- The teams behind Streamlit, scikit-learn, Plotly, and pandas
- Open-source ML/data science community for tools, resources, and inspiration


---

*Project Date: July 2025*
