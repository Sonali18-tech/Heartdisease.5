# Decision Trees and Random Forests - Heart Disease Prediction

This task demonstrates how to implement and evaluate tree-based machine learning models using the **Heart Disease dataset**.

---

##  Introduction

### Decision Trees
A **Decision Tree** is a supervised machine learning algorithm used for classification and regression. It splits the data into subsets based on feature values using metrics like **Gini Index** or **Information Gain (Entropy)**.

### Random Forests
A **Random Forest** is an ensemble of decision trees. It improves accuracy and reduces overfitting by:
- Training multiple trees on random subsets of data and features (Bagging)
- Aggregating predictions (majority vote for classification)

---

## Dataset Description

**Dataset Used:** [Heart Disease Dataset from Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Columns include:**
- Age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), ECG results (restecg), max heart rate (thalach), exercise-induced angina (exang), oldpeak, slope, ca, thal, and target.

**Target:**  
- `1` = Presence of heart disease  
- `0` = No heart disease

---

## Key Results

| Model            | Accuracy |
|------------------|----------|
| Decision Tree    | ~83%     |
| Random Forest    | ~89%     |

- **Best Parameters:** `max_depth=4` improved generalization and reduced overfitting.
- **Top Features:** `cp`, `thalach`, `oldpeak`, `ca`, `thal`

---

##  Code Structure Summary

```plaintext
├── heart.csv                  # Dataset
├── decision_tree_rf.ipynb     # Main python script on spyder
├── tree_visualization.png     # Visual representation of the decision tree
├── feature_importance.png     # Bar chart of feature importance
└── README.md                  # This documentation file

##Visuals Included
Decision Tree plot using sklearn.tree.plot_tree
Feature Importance bar plot using seaborn

## Libraries Used
pandas, numpy — data manipulation
scikit-learn — model building, evaluation
matplotlib, seaborn — data visualization
graphviz (optional) — tree visualization

Author
Sonali18-tech

