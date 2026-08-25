# Employee Attrition Prediction & Retention Strategy

A machine learning project that predicts employee attrition and translates the results into a data-driven retention strategy business proposal.

## Overview

Employee turnover is costly — replacing an employee in the UK costs an average of £25,000, rising to £100,000 for senior roles. Most organizations only learn why employees leave *after* they've already gone, through exit interviews or anonymous surveys. This project takes a proactive approach: using historical HR data to predict which employees are at risk of leaving *before* they do, so retention efforts can be targeted rather than reactive.

## Dataset

- **File:** `dataset.xlsx`
- **Records:** 1,470 employees
- **Target variable:** `Attrition` (Yes/No)
- **Class distribution:** 1,232 stayed (83.8%) / 238 left (16.2%) — an imbalanced classification problem
- **Features:** 40+ attributes covering demographics (Age, Gender, MaritalStatus), job characteristics (Department, JobRole, JobLevel, OverTime), compensation (MonthlyIncome, Incentive, StockOptionLevel), satisfaction/wellbeing (JobSatisfaction, WorkLifeBalance, StressRating, EnvironmentSatisfaction), and tenure (YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion)

Constant/identifier columns (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`, `Year`) were dropped prior to modeling as they carry no predictive signal.

## Approach

1. **Preprocessing**
   - Encoded target variable (`Attrition`: Yes → 1, No → 0)
   - One-hot encoded categorical features
   - Stratified 70/30 train-test split to preserve class balance across splits
   - Standardized features (mean = 0, std = 1)

2. **Modeling**
   Three classifiers were tuned via `GridSearchCV` (5-fold cross-validation):
   - Logistic Regression
   - K-Nearest Neighbors
   - Gradient Boosting Classifier

3. **Evaluation**
   Models were compared on accuracy, precision, F1-score, and ROC-AUC — ROC-AUC in particular, since the target class is imbalanced and accuracy alone can be misleading.

4. **Feature importance**
   Extracted from the best-performing model to identify which factors most influence attrition, informing the retention strategy recommendations.

## Results

| Model | Accuracy | Precision | F1-Score | ROC-AUC | Best Parameters |
|---|---|---|---|---|---|
| Logistic Regression | 86.39% | 65.71% | 43.40% | 0.8459 | `C=0.1, penalty='l2'` |
| K-Nearest Neighbors | 83.67% | 45.45% | 12.20% | 0.6920 | `n_neighbors=11, weights='uniform'` |
| **Gradient Boosting Classifier** | **87.53%** | **70.00%** | **50.45%** | **0.8497** | `learning_rate=0.2, n_estimators=200` |

Gradient Boosting Classifier was selected as the final model, achieving the highest accuracy, precision, F1-score, and ROC-AUC.

**Top 5 most important features** (Gradient Boosting):
1. Incentive (0.0964)
2. StressRating (0.0888)
3. MonthlyIncome (0.0818)
4. JobLevel (0.0670)
5. Age (0.0668)

## Business Recommendations

Based on feature importance, two intervention categories were proposed:

- **Financial:** Flag high performers who are underpaid relative to peers for proactive salary review; conduct annual incentive reviews to keep pay competitive.
- **Well-being:** Offer flexible work arrangements to address stress-related attrition risk; further investigate age-related attrition patterns.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (GridSearchCV, LogisticRegression, KNeighborsClassifier, GradientBoostingClassifier)
- seaborn, matplotlib

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/employee-attrition-prediction.git
   cd employee-attrition-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib openpyxl
   ```

3. **Run the script**
   ```bash
   python code.py
   ```
   Make sure `dataset.xlsx` is in the same directory as the script (or update the file path — the script currently expects `data.csv.xlsx`).

## Project Structure

```
employee-attrition-prediction/
├── code.py             # Data preprocessing, model training, and evaluation
├── dataset.xlsx         # Employee HR dataset
└── README.md
```

## Author

Chun Jia Bao
