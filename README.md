# 🌍 GeoHazard ML

### Geological Hazard Risk Classification using Machine Learning

---

## 📌 Overview

GeoHazard ML is a **supervised machine learning system** that predicts the **geological hazard risk level** of any location using seismic (earthquake) data.

The model classifies locations into five categories:
**Very Low, Low, Medium, High, Very High**

It demonstrates that **seismic activity alone can approximate global risk patterns**, achieving high predictive performance.

---

## 🎯 Problem Statement

Global risk systems like the INFORM Risk Index use **50+ socio-economic indicators** and operate only at **country level**.

This project addresses:

> *Can we predict hazard risk using only seismic data at a finer spatial resolution?*

---

## 🚀 Key Contributions

* ✅ Built a **grid-based global dataset (1°×1° resolution)**
* ✅ Combined **USGS Earthquake data + INFORM Risk Index**
* ✅ Engineered **18 meaningful features** from raw seismic data
* ✅ Handled **severe class imbalance (28:1)** using SMOTE
* ✅ Achieved **80.5% accuracy using XGBoost**
* ✅ Developed **interactive UI (Flask + Gradio + Streamlit)**
* ✅ Enabled **what-if analysis for seismic changes**

---

## 📊 Dataset

### 🔹 Source 1: USGS Earthquake Catalog

* 166,000+ earthquake events
* Time range: **2000–2023**
* Features: latitude, longitude, magnitude, depth

### 🔹 Source 2: INFORM Risk Index

* 191 countries
* Risk score (0–10)
* Converted into 5 categories

---

## 🧠 Data Pipeline

1. **Data Collection**
2. **Data Cleaning**
3. **Grid Creation (1°×1°)**
4. **Aggregation of earthquake data**
5. **Spatial Join (grid → country)**
6. **Label Assignment (INFORM risk)**
7. **Feature Engineering (18 features)**
8. **Preprocessing (Scaling + SMOTE)**
9. **Model Training & Evaluation**

---

## 🔍 Exploratory Data Analysis (EDA)

* Severe class imbalance detected
* Weak linear correlation between features and risk
* Strong geographic patterns (Ring of Fire regions)
* Overlapping feature distributions → non-linear modeling required

---

## ⚙️ Feature Engineering

### 🔹 Core Features

* Earthquake count
* Mean magnitude
* Mean depth

### 🔹 Engineered Features

* Frequency score (log transform)
* Seismic energy proxy
* Magnitude-depth ratio
* Neighbor earthquake count

### 🔹 Geographic Features

* Latitude
* Continent

---

## 🧪 Models Used

| Model               | Purpose                        |
| ------------------- | ------------------------------ |
| Logistic Regression | Baseline                       |
| Random Forest       | Non-linear ensemble            |
| XGBoost             | Final model (best performance) |

---

## 📈 Model Performance

| Model               | Accuracy  | F1 Score | ROC-AUC   |
| ------------------- | --------- | -------- | --------- |
| Logistic Regression | 58.5%     | 0.48     | 0.906     |
| Random Forest       | 76.7%     | 0.65     | 0.959     |
| **XGBoost**         | **80.5%** | **0.68** | **0.961** |

---

## 🏆 Final Model

> **XGBoost** selected due to highest accuracy and best handling of complex patterns and imbalance.

---

## 🌐 Deployment

The system is deployed using:

* 🔹 Flask (Render)
* 🔹 Gradio (Hugging Face Spaces)
* 🔹 Streamlit

### Features:

* Risk prediction interface
* Interactive global map
* Confidence score output
* What-if scenario simulation

---

## ⚠️ Limitations

* Lower performance for **Very Low risk class**
* Uses only seismic data (no socio-economic factors)
* Country-level labels applied to grid cells
* No real-time updating (requires retraining)

---

## 🔮 Future Scope

* Add socio-economic features (GDP, population)
* Use temporal modeling (LSTM)
* Increase spatial resolution
* Add explainability (SHAP)
* Automate real-time updates

---

## 🧠 Tech Stack

* Python
* Scikit-learn
* XGBoost
* Pandas, NumPy
* GeoPandas
* Plotly
* Flask / Gradio / Streamlit

---

## 📌 Conclusion

GeoHazard ML successfully demonstrates that:

> “Seismic data, combined with machine learning, can effectively approximate complex global risk systems with high accuracy.”

---

## 👨‍💻 Authors

* Nithish V
* Bhuvanesh A

---

## 📜 License

This project is for academic and research purposes.

---
