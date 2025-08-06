# 🔋 Enhancing Battery Performance: AI-Driven State Level Prediction for Electric Vehicles

This project focuses on predicting the Remaining Useful Life (RUL) of electric vehicle (EV) batteries using machine learning techniques. Built with Python and deployed using Streamlit, the system analyzes battery cycle and voltage data to estimate battery health and forecast maintenance needs.

---

## 🚀 Overview

Battery performance is crucial for the reliability and efficiency of electric vehicles. This project uses supervised learning techniques, particularly Random Forest Regression, to predict how many more cycles a battery can go through before needing replacement.

The web application is built using Streamlit, allowing users to upload battery data and receive real-time RUL predictions through an interactive interface.

---

## ✅ Key Features

- Upload your own battery dataset (CSV format)
- Predict Remaining Useful Life (RUL) of batteries
- Visualize actual vs predicted values
- Compare multiple regression models
- Reuse trained models with Pickle
- Streamlit-based clean and responsive UI

---

## 🧠 Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn** (RandomForestRegressor, Gradient Boosting, KNN, etc.)
- **Streamlit** – Web UI and deployment
- **Pickle** – Saving and loading ML models
- **Matplotlib, Seaborn** – For plotting model performance

---

## 🖥️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ev-battery-prediction.git
cd ev-battery-prediction
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

### 4. Upload Dataset
- Upload `Battery_RUL.csv` (or your own dataset)
- Click “Predict” to view RUL predictions
- View graphs and model evaluation metrics (MSE, RMSE)

---

## 📁 Folder Structure
```
ev-battery-prediction/
├── streamlit_app.py
├── Battery_RUL.csv
├── model.pkl
├── requirements.txt
├── assets/ (optional)
│   └── project_recording.mp4
└── README.md
```

---

## 📊 Models Used

- Random Forest Regressor ✅
- Gradient Boosting Regressor
- AdaBoost Regressor
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- Extra Trees Regressor

**Evaluation Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
