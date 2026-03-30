# 🏨 Hotel Booking Cancellation Prediction

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Predicting hotel booking cancellations using machine learning to help INN Hotels Group minimize revenue loss and optimize room occupancy.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Business Impact](#business-impact)
- [Skills](#skills)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [Repository Structure](#repository-structure)
- [Author](#author)

---

## Project Overview

This project addresses a critical business challenge faced by INN Hotels Group in Portugal — the high rate of booking cancellations that directly erodes revenue, inflates distribution costs, and disrupts operational planning. By leveraging machine learning on historical booking data, this project identifies the key drivers behind cancellations and builds a predictive model capable of flagging at-risk bookings in advance.

The solution empowers hotel management to intervene proactively — adjusting cancellation policies, offering targeted incentives, and managing room inventory with greater confidence.

👉 [Open the notebook to explore full analysis](notebook/Hotel_Cancellation_Prediction_Model.ipynb)

---

## Business Problem

### Real-World Context

A significant share of hotel bookings are called off due to cancellations or no-shows. The widespread adoption of online booking platforms has further amplified this problem by making it easier than ever for customers to book and cancel at little or no cost.

The financial and operational repercussions for INN Hotels Group include:

- **Lost Revenue** — Rooms that cannot be resold after a last-minute cancellation.
- **Increased Distribution Costs** — Higher commissions paid to channels to fill vacated rooms.
- **Margin Erosion** — Forced last-minute price drops to resell cancelled inventory.
- **Operational Disruption** — Misallocation of housekeeping, staffing, and F&B resources.

### Stakeholders

- Hotel Revenue Management teams
- Operations and Front Office departments
- Executive leadership responsible for profitability

### Decision Impact

A reliable cancellation prediction model enables the hotel to take pre-emptive action — from applying dynamic cancellation fees to prioritizing loyalty-building efforts — ultimately improving occupancy rates and protecting profit margins.

---

## Dataset

| Attribute | Detail |
|---|---|
| **Source** | INN Hotels Group, Provided as part of PGP-DSBA project |
| **Size** | 36,275 rows × 19 columns |
| **Missing Values** | None |
| **Duplicate Values** | None |
| **Target Variable** | `booking_status` (Canceled / Not_Canceled) |
| **Class Distribution** | 67.2% Not Canceled · 32.8% Canceled |

### Key Features

| Feature | Type | Description |
|---|---|---|
| `lead_time` | Numerical | Days between booking date and arrival date |
| `avg_price_per_room` | Numerical | Average room price per night (in euros) |
| `no_of_special_requests` | Numerical | Number of special requests made by the guest |
| `market_segment_type` | Categorical | Booking channel (Online, Offline, Corporate, Aviation, Complementary) |
| `repeated_guest` | Binary | Whether the guest is a returning customer |
| `arrival_month` | Numerical | Month of arrival |
| `type_of_meal_plan` | Categorical | Meal plan selected by the guest |
| `room_type_reserved` | Categorical | Type of room reserved (encoded) |
| `no_of_adults` | Numerical | Number of adults in the booking |
| `no_of_previous_cancellations` | Numerical | Prior cancellations by the same guest |

**Data Types:** 14 numerical columns, 5 categorical columns (int64, float64, object)

---

## Methodology

### 1. Data Understanding

The dataset was loaded and thoroughly inspected to understand its structure and quality. The dataset contains **36,275 records** and **19 features** with **no missing, null, or duplicate values**. Statistical summaries were computed for both numerical and categorical variables to establish baseline distributions. Key initial findings included:

- The average lead time is approximately **85 days** (range: 0–443 days), indicating high variability in advance booking behavior.
- The average room price is **€103.42**, ranging from €0 to €540 — reflecting a dynamic pricing environment.
- Only **2.6% of guests are repeat visitors**, highlighting a significant customer retention challenge.
- The `Booking_ID` column was identified as non-informative (unique per row) and flagged for removal.

---

### 2. Exploratory Data Analysis

#### Univariate Analysis

Each variable was analyzed independently to understand its distribution, frequency, and range:

- **`avg_price_per_room`**: Heavily right-skewed with high-value outliers representing premium/luxury bookings — retained as is.
- **`lead_time`**: Also right-skewed; outliers represent far-advance bookings and are contextually valid — retained.
- **`no_of_adults`**: 72% of bookings involve 2 adults, suggesting couples dominate the guest mix.
- **`no_of_children`**: 92.6% of bookings involve no children — primary demographic is couples or solo travelers.
- **`no_of_weekend_nights`**: 46.5% of stays involve zero weekend nights, with 1–2 night weekend stays being most common.
- **`no_of_week_nights`**: 2-night weekday stays are the most frequent (31.5%), followed by 1 night (26.2%) and 3 nights (21.6%).
- **`required_car_parking_space`**: 96.9% of guests do not require parking — most guests arrive via public transport or from distant locations.
- **`arrival_year`**: 82% of bookings fall in 2018 vs. 18% in 2017 — significant business growth year-over-year.
- **`arrival_month`**: October is the busiest month at 14.7%, followed by September (12.7%) and August (10.5%).
- **`repeated_guest`**: Only 2.6% of guests are repeat visitors — a critical area for strategic improvement.
- **`no_of_special_requests`**: 54.5% of guests made no special requests; 31.4% made exactly one.
- **`type_of_meal_plan`**: Meal Plan 1 (Breakfast) dominates at 76.7%.
- **`room_type_reserved`**: Room Type 1 accounts for 77.5% of all bookings.
- **`market_segment_type`**: Online bookings lead at 64%, followed by Offline at 29%.
- **`booking_status`**: **32.8% of all bookings were canceled** — nearly one-third of total bookings result in cancellation.

#### Bivariate Analysis

The relationships between individual features and the target variable (`booking_status`) were examined systematically:

- **Correlation Heatmap (Numerical Variables):** Identified moderate correlations such as `no_of_previous_cancellations` and `no_of_previous_bookings_not_canceled` (r = 0.54), and `avg_price_per_room` with `no_of_adults` (r = 0.30). Most other pairs showed near-zero correlations.

- **`market_segment_type` vs `avg_price_per_room`:** Online customers pay the highest average room price (€112.26), followed by Offline (€91.63), Corporate (€82.91), and Complementary (€3.14). Hotel pricing is tightly coupled to the booking channel.

- **`repeated_guest` vs `booking_status`:** Repeated guests have a near-negligible cancellation rate of just **1.72%**, compared to 33.58% for first-time guests — underscoring the immense value of customer loyalty.

- **`no_of_special_requests` vs `booking_status`:** As the number of special requests increases from 0 to 5, the proportion of cancellations consistently decreases. Guests with zero special requests exhibit a cancellation rate exceeding 50%.

- **`market_segment_type` vs `booking_status`:** Online channel accounts for **71.3% of all cancellations**, followed by Offline at 26.52%.

- **`arrival_month` vs `booking_status`:** October records the highest absolute cancellations (1,880), consistent with it being the busiest month.

- **`avg_price_per_room` vs `booking_status`:** Canceled bookings exhibit a wider price distribution and a higher median price (~€100) compared to non-canceled bookings (~€80), confirming price sensitivity as a cancellation driver.

- **`lead_time` vs `booking_status`:** Canceled bookings show a significantly higher median lead time and broader IQR (60–250 days) vs. non-canceled bookings (20–100 days). Longer advance bookings are substantially more likely to be canceled.

---

### 3. Data Preprocessing

- **Outlier Treatment:** After visual inspection using boxplots across all 14 numerical variables, no outliers required removal. Outliers in `lead_time` and `avg_price_per_room` were confirmed as genuine business values (long-advance bookings and premium rooms respectively).
- **Target Encoding:** `booking_status` was encoded from categorical to binary integer (`Not_Canceled = 0`, `Canceled = 1`).
- **Feature Removal:** `Booking_ID` was dropped as it carries no predictive value.
- **Categorical Encoding:** Categorical variables were one-hot encoded for model compatibility.
- **Train-Test Split:** Data was divided into training (70%) and test (30%) sets — resulting in 25,392 training records and 10,883 test records — with class proportions preserved across both splits.

---

### 4. Model Building

Four classification models were built and evaluated:

**1. Logistic Regression (statsmodels)**
A baseline logistic regression was built using statsmodels to obtain coefficient significance via p-values. Initial inspection revealed high p-value variables (e.g., `type_of_meal_plan_Meal Plan 3`, `market_segment_type_Online`) potentially affected by multicollinearity.

**2. Naïve Bayes Classifier**
A Gaussian Naïve Bayes model was trained as an alternative probabilistic baseline.

**3. K-Nearest Neighbors Classifier (K=2)**
KNN was implemented starting with K=2 as a distance-based non-parametric approach, with recall used as the optimization criterion.

**4. Decision Tree Classifier**
A CART-based Decision Tree was built to leverage its interpretability and feature importance ranking capabilities.

---

### 5. Model Performance Improvement

**Logistic Regression — Multicollinearity & Threshold Tuning:**
- Variance Inflation Factors (VIF) were computed for all predictors. `market_segment_type_Online` (VIF = 69.47) and `market_segment_type_Offline` (VIF = 62.51) exhibited severe multicollinearity.
- `market_segment_type_Online` was dropped, after which all VIF values fell below 5 — confirming resolution of multicollinearity.
- Insignificant features (p-value > 0.05) were iteratively removed through a stepwise elimination loop until all remaining features were statistically significant.
- The optimal classification threshold was determined using the **ROC Curve**, yielding an **AUC of 0.86** and an **optimal threshold of 0.333**.

**KNN — Hyperparameter Tuning:**
- K values from 2 to 20 were evaluated using recall as the selection criterion.
- The **optimal K = 3** was identified, achieving a recall of 0.7372 on the validation set.

**Decision Tree — Pre-Pruning:**
- The base decision tree was observed to overfit severely (training recall = 0.98, test recall = 0.79).
- Pre-pruning constraints (maximum depth, minimum samples per split/leaf) were applied to reduce model complexity and improve generalization.

---

### 6. Model Evaluation

All models were evaluated on Accuracy, Recall, Precision, and F1 Score across both training and test sets. Recall was prioritized as the primary metric, since **False Negatives** (predicting a booking will not be canceled when it actually will) carry a higher business cost than False Positives.

| Model | Test Accuracy | Test Recall | Test Precision | Test F1 |
|---|---|---|---|---|
| Logistic Regression (Base) | 0.807 | 0.633 | 0.740 | 0.682 |
| Logistic Regression (Tuned) | 0.778 | 0.766 | 0.634 | 0.694 |
| Naïve Bayes | 0.407 | 0.964 | 0.352 | 0.516 |
| KNN (Base, K=2) | 0.844 | 0.632 | 0.855 | 0.727 |
| KNN (Tuned, K=3) | 0.844 | 0.737 | 0.776 | 0.756 |
| Decision Tree (Base) | 0.865 | 0.795 | 0.794 | 0.795 |
| **Decision Tree (Tuned)** | **0.854** | **0.820** | **0.756** | **0.787** |

> **Final Model Selected: Tuned Decision Tree** — best balance of recall, precision, and F1 across both training and test sets, with minimal overfitting.

---

## Key Results

- **32.8%** of all bookings in the dataset were canceled — a major business risk.
- **Lead time** is the single most important predictor of cancellation (confirmed by both feature importance and the first decision tree split at ≤ 151.5 days).
- **Online channel** accounts for 71.3% of all cancellations — despite generating the highest revenue per booking.
- **Repeat guests** cancel at just 1.72% — 19× lower than first-time guests.
- **Guests with special requests** are consistently less likely to cancel — each additional request correlates with meaningfully lower cancellation probability.
- **Higher room prices** (above ~€100) are associated with increased cancellation likelihood.
- The **Tuned Decision Tree** achieved a test recall of **0.82**, accuracy of **0.854**, and F1 score of **0.787** — the strongest overall performance across all evaluated models.
- The top 4 features by importance: `lead_time`, `market_segment_type_Online`, `no_of_special_requests`, `avg_price_per_room`.

---

## Business Impact

**1. Implement Lead-Time Based Cancellation Policies**
Bookings with lead times exceeding 150 days should be subject to tiered, non-refundable deposits or dynamic cancellation fees. Since lead time is the single strongest predictor of cancellation, this policy directly targets the highest-risk bookings and protects revenue.

**2. Prioritize Online Channel Management**
With 71.3% of cancellations originating from online channels, the hotel should implement stricter online cancellation terms, invest in loyalty-based incentives for online bookers, and consider requiring partial prepayment for high-lead-time online reservations.

**3. Launch a Structured Guest Loyalty Program**
Repeat guests cancel at only 1.72%. Developing a formal rewards program — including priority check-in, room upgrades, and personalized offers — can meaningfully increase repeat visit rates and reduce the overall cancellation risk profile of the booking mix.

**4. Personalize Engagement to Drive Special Requests**
Guests who make special requests are measurably less likely to cancel. The hotel should actively solicit guest preferences during the booking journey (e.g., floor preference, dietary needs, pillow type) to increase engagement and booking commitment.

**5. Integrate the Tuned Decision Tree into Revenue Operations**
The model should be embedded into the hotel's booking management system to generate real-time cancellation risk scores for each new reservation. Revenue managers can use these scores to prioritize overbooking strategies, apply dynamic pricing, and trigger proactive outreach campaigns before high-risk bookings lapse.

---

## Skills

### Technical Skills

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-77AC1D?style=for-the-badge&logo=seaborn&logoColor=white)
![Logistic Regression](https://img.shields.io/badge/Logistic_Regression-4051B5?style=for-the-badge)
![Decision Tree](https://img.shields.io/badge/Decision_Tree-4051B5?style=for-the-badge)
![KNN](https://img.shields.io/badge/KNN-4051B5?style=for-the-badge)
![Naive Bayes](https://img.shields.io/badge/Naive_Bayes-4051B5?style=for-the-badge)
![Classification](https://img.shields.io/badge/Classification-4051B5?style=for-the-badge)
![EDA](https://img.shields.io/badge/EDA-FFA500?style=for-the-badge)
![Feature Engineering](https://img.shields.io/badge/Feature_Engineering-FFA500?style=for-the-badge)
![VIF](https://img.shields.io/badge/VIF-FFA500?style=for-the-badge)
![ROC/AUC](https://img.shields.io/badge/ROC--AUC-4051B5?style=for-the-badge)
![Confusion Matrix](https://img.shields.io/badge/Confusion_Matrix-4051B5?style=for-the-badge)
![Hyperparameter Tuning](https://img.shields.io/badge/Hyperparameter_Tuning-4051B5?style=for-the-badge)
![Supervised Learning](https://img.shields.io/badge/Supervised_Learning-20B2AA?style=for-the-badge)
![Business Analytics](https://img.shields.io/badge/Business_Analytics-20B2AA?style=for-the-badge)
![Hospitality Analytics](https://img.shields.io/badge/Hospitality_Analytics-20B2AA?style=for-the-badge)

### Soft Skills

![Business Acumen](https://img.shields.io/badge/Business_Acumen-4B0082?style=for-the-badge)
![Storytelling with Data](https://img.shields.io/badge/Storytelling_with_Data-25D366?style=for-the-badge)
![Executive Communication](https://img.shields.io/badge/Executive_Communication-FF4500?style=for-the-badge)
![Analytical Thinking](https://img.shields.io/badge/Analytical_Thinking-00CED1?style=for-the-badge)
![Problem Solving](https://img.shields.io/badge/Problem_Solving-800080?style=for-the-badge)

---

## Key Learnings

- **Lead time is the most powerful cancellation signal.** The decision tree's first split at 151.5 days validates that customers who book far in advance are structurally more uncertain — a finding with direct policy implications.
- **Multicollinearity can silently distort logistic regression results.** Computing VIF before trusting p-values is a non-negotiable step in regression-based workflows. Dropping `market_segment_type_Online` (VIF = 69.47) transformed the reliability of coefficient estimates.
- **Recall must be the guiding metric when false negatives are costly.** Optimizing threshold using the ROC curve shifted the Logistic Regression recall from 0.63 to 0.76 — a direct business win despite modest F1 improvement.
- **Pre-pruning is essential for decision trees on real-world tabular data.** The base tree overfit severely; structured pre-pruning brought training and test performance into close alignment without sacrificing meaningful recall.
- **Customer behavior signals matter more than demographic data.** Features like `no_of_special_requests` and `repeated_guest` — representing intent and commitment — outperformed raw demographic features in predictive power.
- **EDA is not optional — it is foundational.** Discoveries like the inverse relationship between special requests and cancellations, and the outsized price sensitivity among canceled bookings, emerged only through careful bivariate analysis.

---

## Future Improvements

1. **Ensemble Methods:** Implement Random Forest, Gradient Boosting (XGBoost, LightGBM), and stacking ensembles to push recall and F1 beyond the Decision Tree baseline, while maintaining interpretability.
2. **Class Imbalance Handling:** Apply SMOTE (Synthetic Minority Oversampling Technique) or class-weight adjustments to explicitly address the 67:33 class split and further reduce false negatives.
3. **Feature Engineering:** Construct derived features such as `price_per_adult`, `total_nights` (weekend + weekday), `booking_season`, and `cancellation_history_ratio` to enrich model inputs.
4. **Cross-Validation & Hyperparameter Optimization:** Replace single train-test splits with stratified k-fold cross-validation and GridSearchCV/Optuna-based hyperparameter tuning for more robust model selection.
5. **Real-Time Deployment Pipeline:** Package the final model into a REST API (Flask/FastAPI) or integrate with the hotel's Property Management System (PMS) to generate live cancellation risk scores at the point of booking.

---

## Repository Structure

```
hotel-booking-cancellation-prediction-model/
│
├── data/
│   └── INNHotelsGroup.csv          # Original dataset
│
├── notebook/
│   └── Hotel_Cancellation_Prediction_Model.ipynb # Analysis notebook
│
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
└── .gitignore              # Git ignore file
```


## Author
| **Nabankur Ray** |
| :--- |
| Passionate about real-world data-driven solutions |
| [![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/nabankur14) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/nabankur-ray-876582181/) |


![GitHub Stats](https://github-readme-stats-eight-theta.vercel.app/api?username=nabankur14&show_icons=true)

---

⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!