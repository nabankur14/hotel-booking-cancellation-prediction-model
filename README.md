# 🏨 Hotel Booking Cancellation Prediction

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

> **Predicting hotel booking cancellations to minimize revenue loss and optimize inventory management.**

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Results](#-key-results)
- [Business Impact](#-business-impact)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Project Overview
This project aims to build a machine learning solution for **INN Hotels Group** to predict booking cancellations. By analyzing customer data, we identify key factors driving cancellations and provide a predictive model to help the hotel chain take proactive measures.

## 💼 Business Problem
**The Challenge:**
The hotel industry faces significant revenue loss due to last-minute cancellations and no-shows. When a guest cancels, the hotel loses potential revenue if the room cannot be resold.

**The Goal:**
- Predict which bookings are likely to be canceled.
- Understand the drivers behind cancellations (e.g., lead time, market segment).
- Formulate profitable cancellation policies.

**Stakeholders:**
- Hotel Management (Revenue Managers)
- Marketing Team
- Operations Team

## 📊 Dataset
The dataset contains **36,275 observations** and **19 variables** representing booking details.

- **Source:** INN Hotels Group (Provided)
- **Size:** 36,275 rows, 19 columns
- **Key Features:**
  - `lead_time`: Days between booking and arrival.
  - `avg_price_per_room`: Dynamic pricing of the room.
  - `no_of_special_requests`: Count of special requests made.
  - `market_segment_type`: How the booking was made (Online, Offline, etc.).
  - `booking_status`: Target variable (Canceled / Not Canceled).

## ⚙️ Methodology

1.  **Data Cleaning & EDA**:
    - Handled missing values (none found).
    - Analyzed distributions and correlations.
    - Identified outliers in `lead_time` and `avg_price_per_room`.
2.  **Preprocessing**:
    - Label Encoding for target variable (`Not_Canceled`: 0, `Canceled`: 1).
    - One-Hot Encoding for categorical features (`market_segment_type`, `meal_plan`, etc.).
    - Scaling using `StandardScaler`.
3.  **Model Building**:
    - **Logistic Regression**: Baseline model for interpretability.
    - **K-Nearest Neighbors (KNN)**: Non-parametric approach.
    - **Decision Tree**: Tree-based model for capturing non-linear relationships.
4.  **Hyperparameter Tuning**:
    - Used `GridSearchCV` to optimize the Decision Tree (optimizing for **Recall**).
5.  **Evaluation**:
    - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    - Focus on **Recall** to minimize False Negatives (predicting "Not Canceled" when it is actually "Canceled").

## 📈 Key Results

| Model | Accuracy | Recall (Test) | Precision | F1-Score |
|-------|----------|---------------|-----------|----------|
| Logistic Regression | ~80% | ~60% | ~75% | ~66% |
| **Tuned Decision Tree** | **~87%** | **~78%** | **~77%** | **~77%** |

*Note: The Tuned Decision Tree achieved the best balance, significantly improving the identification of canceled bookings.*

![Confusion Matrix](visuals/confusion_matrix_tuned_dt.png)
*(Run the notebook to generate this visual)*

## 🚀 Business Impact
1.  **Dynamic Cancellation Policies**: Implement stricter cancellation fees for "High Risk" bookings identified by the model (e.g., high lead time, online segment).
2.  **Overbooking Strategy**: Use predictions to safely overbook, ensuring full occupancy even with cancellations.
3.  **Targeted Marketing**: Offer incetives (discounts, upgrades) to high-risk customers to encourage them to keep their booking.

## 🛠 Tech Stack
**Technical Skills:**
- **Language**: Python 3.8+
- **Libraries**: Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, Git

**Soft Skills:**
- Problem Solving
- Business Communication
- Storytelling with Data

## 📂 Repository Structure

```
project-name/
│
├── data/
│   ├── raw/                # Original dataset (INNHotelsGroup.csv)
│   └── processed/          # Processed data (optional)
│
├── notebooks/
│   ├── analysis.ipynb      # Main analysis notebook (Refactored)
│   └── archived_model.ipynb # Original legacy notebook
│
├── src/
│   ├── data_preprocessing.py # Data loading & cleaning
│   ├── modeling.py           # Model training & tuning logic
│   └── evaluation.py         # Metrics & plotting functions
│
├── reports/
│   └── Booking...Report.pdf # Business report
│
├── visuals/                # Generated plots and charts
│
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitignore
```

## 🏃 How to Run
1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd hotel-booking-cancellation-prediction
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis**:
    ```bash
    jupyter notebook notebooks/analysis.ipynb
    ```

## 🔮 Future Improvements
1.  **Feature Engineering**: Create new features like `total_nights` or `seasonality_index` to capture more patterns.
2.  **Advanced Models**: Experiment with Ensemble methods like **Random Forest** or **XGBoost** for potentially higher accuracy.
3.  **Deployment**: Deploy the model as an API using Flask/FastAPI or creating a Streamlit dashboard for real-time predictions.
4.  **Cost-Benefit Analysis**: Quantify the financial gain of using the model vs. current baseline.

## ✍️ Author
| **Nabankur Ray** |
| :--- |
| **Data Scientist | Business Analyst | Machine Learning Engineer** |
| [![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/nabankur14) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://linkedin.com/in/nabankur14) |
