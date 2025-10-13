<h1 align="center" style="color:#2b7a78;">Hotel Booking Cancellation Prediction – INN Hotels Group</h1>
<h3 align="center" style="color:#17252a;">Predicting Booking Cancellations Using Machine Learning to Optimize Revenue and Customer Retention</h3>

<p align="center">
  <strong>Author:</strong> <a href="https://github.com/nabankur14" target="_blank" style="color:#3aafa9;">Nabankur Ray</a>  
</p>

<hr>

<h2 style="color:#17252a;">Overview</h2>
<p>
This project leverages <strong>Machine Learning</strong> and <strong>Data Analytics</strong> to predict hotel booking cancellations for 
<strong>INN Hotels Group</strong> in Portugal. The objective is to minimize revenue loss, improve occupancy forecasting, and 
enhance customer retention by identifying patterns and building a predictive system that flags potential cancellations.  
Comprehensive <em>Exploratory Data Analysis (EDA)</em>, <em>feature engineering</em>, and <em>model optimization</em> were performed 
to extract actionable business insights.
</p>

<details open>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Objective</summary>
  <p>
  The primary goals of this project are to:
  <ul>
    <li>Analyze key drivers influencing booking cancellations across different customer segments.</li>
    <li>Build predictive ML models to forecast whether a booking will be canceled.</li>
    <li>Compare model performances (Logistic Regression, Naive Bayes, KNN, Decision Tree) and select the best one.</li>
    <li>Provide business strategies to reduce cancellations and improve customer retention.</li>
    <li>Enable proactive decision-making through data-driven policy design for refunds and dynamic pricing.</li>
  </ul>
  </p>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Dataset</summary>
  <ul>
    <li><strong>Source:</strong> INN Hotels Group internal booking database (Portugal).</li>
    <li><strong>Size:</strong> 36,275 rows × 19 columns.</li>
    <li><strong>Features:</strong>
      <ul>
        <li><code>no_of_adults</code> – Number of adults per booking</li>
        <li><code>no_of_children</code> – Number of children per booking</li>
        <li><code>lead_time</code> – Days between booking and arrival</li>
        <li><code>avg_price_per_room</code> – Average price per night (in Euros)</li>
        <li><code>type_of_meal_plan</code> – Chosen meal plan (e.g., Breakfast, Half-board)</li>
        <li><code>market_segment_type</code> – Booking channel (Online, Offline, Corporate, etc.)</li>
        <li><code>no_of_special_requests</code> – Total special requests made by the guest</li>
        <li><code>booking_status</code> – Target variable (Canceled / Not Canceled)</li>
      </ul>
    </li>
    <li><strong>Data Quality:</strong> No missing or duplicate values. Mixed numerical and categorical data cleaned and validated.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Methodology</summary>
  <ol>
    <li><strong>Data Cleaning:</strong> Checked for null, duplicate, and outlier values; dropped irrelevant columns (e.g., Booking_ID).</li>
    <li><strong>Exploratory Data Analysis (EDA):</strong> Used histograms, boxplots, heatmaps, and bar charts to uncover cancellation trends and customer behavior.</li>
    <li><strong>Feature Engineering:</strong> 
      <ul>
        <li>Converted categorical variables to numerical (Label Encoding & Dummies).</li>
        <li>Handled multicollinearity using <strong>Variance Inflation Factor (VIF)</strong>.</li>
        <li>Split data into train–test sets (70:30).</li>
      </ul>
    </li>
    <li><strong>Model Building & Evaluation:</strong> Implemented four models:
      <ul>
        <li>Logistic Regression (Baseline)</li>
        <li>Naive Bayes Classifier</li>
        <li>K-Nearest Neighbors (KNN)</li>
        <li>Decision Tree Classifier</li>
      </ul>
    </li>
    <li><strong>Model Tuning:</strong> 
      <ul>
        <li>Removed high VIF features to address multicollinearity.</li>
        <li>Optimized Decision Tree via pre-pruning and hyperparameter tuning.</li>
        <li>Used ROC Curve and threshold optimization for Logistic Regression.</li>
      </ul>
    </li>
  </ol>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Tools & Technologies</summary>
  <p>
  <code>Python</code>, <code>Pandas</code>, <code>NumPy</code>, <code>Scikit-learn</code>, <code>Statsmodels</code>,  
  <code>Matplotlib</code>, <code>Seaborn</code>, <code>Jupyter Notebook</code>, <code>Excel</code>
  </p>
</details>

<details open>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Results & Insights</summary>
  <ul>
    <li><strong>Cancellation Rate:</strong> ~33% of bookings were canceled, majorly from online channels.</li>
    <li><strong>Key Predictors:</strong> <code>lead_time</code>, <code>avg_price_per_room</code>, <code>market_segment_type</code>, <code>no_of_special_requests</code>, and <code>repeated_guest</code>.</li>
    <li><strong>Final Model:</strong> Tuned Decision Tree with 85% accuracy, 82% recall, and balanced F1 score.</li>
    <li><strong>Insights:</strong> 
      <ul>
        <li>Higher lead time and price increase cancellation likelihood.</li>
        <li>Repeated guests and those with special requests show low cancellation risk.</li>
        <li>Online bookings contribute to most cancellations — focus needed on retention and engagement.</li>
      </ul>
    </li>
    <li><strong>Business Recommendations:</strong>
      <ul>
        <li>Introduce loyalty programs for repeat customers.</li>
        <li>Implement dynamic cancellation fees for long lead-time bookings.</li>
        <li>Enhance online channel experience and communication.</li>
        <li>Offer flexible pricing during high-cancellation months like October.</li>
      </ul>
    </li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Future Scope</summary>
  <ul>
    <li>Deploy the model via a <strong>Streamlit</strong> app or API for real-time prediction.</li>
    <li>Integrate additional variables like customer reviews or payment type for enhanced accuracy.</li>
    <li>Develop a <strong>Power BI</strong> or <strong>Tableau</strong> dashboard for interactive cancellation trend monitoring.</li>
    <li>Experiment with ensemble models (Random Forest, XGBoost) for further performance gains.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Key Learnings</summary>
  <ul>
    <li>Developed an end-to-end ML workflow from data preprocessing to business strategy formulation.</li>
    <li>Enhanced understanding of <strong>classification modeling, multicollinearity handling,</strong> and <strong>ROC analysis</strong>.</li>
    <li>Learned to bridge technical ML outputs with <strong>business recommendations</strong> for decision-making.</li>
    <li>Strengthened proficiency in <strong>EDA, model evaluation,</strong> and <strong>insight communication</strong>.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Folder Structure</summary>
  <pre style="background:#f0f0f0; padding:10px; border-radius:8px;">

hotel_cancellation_ml_project/
│
├── data/                                      → Raw and processed booking datasets  
├── Nabankur-ML1_Coded_Project.ipynb           → Main Jupyter Notebook (EDA + ML modeling)
├── Nabankur-Business Report - ML1 Coded Project.pdf   → Full business & analytical report
├── README.md                                  → Project documentation (this file)
│
└── results/                                   → Model outputs, visualizations, ROC curves
  </pre>
</details>

<p align="center" style="color:#555;">
>>> All project files are organized and accessible for easy reproducibility and reference.
</p>

<h2 style="color:#17252a;"> #Tags</h2>
<p>
#MachineLearning #DataScience #HotelAnalytics #EDA #Classification #DecisionTree #Python #BusinessIntelligence #PredictiveModeling #CustomerRetention #RevenueOptimization
</p>

<hr>
<p align="center" style="font-size:14px; color:#555;">
© 2025 <strong>Nabankur Ray</strong> | Data Scientist
</p>
