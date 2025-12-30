# IBM HR Analytics: Employee Engagement and Attrition Prediction Project
End-to-end Machine Learning project to predict employee attrition using IBM HR data. Features Logistic Regression, Random Forest, and XGBoost models.

This study is an end-to-end machine learning project developed to protect a company's most valuable asset: its human resources. As part of the project, data from 1,470 employees was analyzed to model attrition trends and create a data-driven roadmap for strategic decision-making.

## Prepared By
**Name**:Nadi Uçar

**Title**: Management Information Systems(MIS) 4th Year Student.

**Project Type**: Tech Istanbul & Ecodation Machine Learning Bootcamp Graduation Project.

**Business-Focused Presentation:** [IBM HR Analytics Attrition Analysis](https://www.canva.com/design/DAG8jTPdb5I/-ckkph_PtJJ3N3ERMUE-wA/edit?utm_content=DAG8jTPdb5I&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Business Problem and Objective (Business Case)
Employee turnover (attrition) means high hiring costs, loss of training investments, and loss of corporate memory.

**Primary Goal**: Predicting an employee's tendency to resign “well before” they pack up their desk.

**Strategic Choice**: In HR analytics, “missing someone who will leave” (False Negative) is much more costly than “saying someone who will stay will leave” (False Positive). Therefore, a Recall (Capture Rate) of 70% is targeted for our model's success.

## Data Preparation and Engineering
The following advanced procedures were applied to improve the model's accuracy:

**Noise Cleaning**: Fixed variables such as **EmployeeCount**, **StandardHours**, and **Over18**, which have no predictive power, were removed.

**Detection of Multicollinearity**: An extremely high correlation of **0.94** was detected between **MonthlyIncome** and **JobLevel**.

To prevent the model from overfitting, JobLevel and repeated financial ratios (**DailyRate**, **MonthlyRate**, **HourlyRate**) were eliminated.

**Robust Scaling**: To avoid losing senior employees (outliers) in the data, RobustScaler, which is resistant to outliers, was used instead of StandardScaler.

**Imbalanced Data Management**: To maintain the “split” ratio in the training and test sets, stratification was performed using the **stratify=y** parameter.

## Modeling and Performance Comparison
Three different algorithms were tested using class weighting methods:

The dataset is split into 80% training and 20% testing.

The imbalance between classes (16% split, 84% remaining) has been strategically managed.

| Model | Precision (İsabet) | Recall (Yakalama) | F1-Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **0.40** | **0.70** | **0.51** |
| Random Forest (Balanced) | 0.40 | 0.09 | 0.14 |
| XGBoost (Weighted) | 0.32 | 0.47 | 0.38 |

**Why Logistic Regression?:** Although it is a linear model, it responded best to weighting, correctly captured 7 out of every 10 people who would leave, and provided “explainable” results thanks to its coefficients.

## Key Findings (Top 10 Factors)
According to the model analysis, the five key factors that most influence employees' decision to leave their jobs and their explanations are as follows:

| Impact Factor (Feature) | Explanation |
| :--- | :--- |
| **Frequent Business Travel** |Frequent travelers are much more likely to leave than infrequent travelers. |
| **Manager Relationship** | The time spent with the current manager directly affects organizational commitment and the decision to stay. |
| **Job Role** | The turnover rate is particularly noticeable among Laboratory Technicians and Sales Representatives. |
| **Marital Status** | The data shows that single employees are more likely to leave their jobs compared to other groups. |
| **Distance from Home** | The increasing distance between the workplace and home is a critical threshold that reduces employee satisfaction. |

**Analysis Note:** These factors have been validated by the coefficients of our Champion Model, Logistic Regression, and the “Top 10 Factors” analysis in our presentation.

## Recommendations for the Business Unit

**Flexible Work**: Remote work options should be offered for those who travel frequently.

**Rotation**: Internal rotation opportunities should be created for employees who have worked with the same manager for a long time.

**Incentives**: Bonus systems for Lab Technicians and Sales Representatives should be revised.

**Early Warning**: The model should be integrated into the HR system to initiate “Stay Interviews.”

# Thank you
I would like to thank my instructor **Deniz Alkan**, who broadened our horizons with his vision throughout the bootcamp process, as well as the **Tech İstanbul** and **Ecodation** teams for providing this opportunity.
