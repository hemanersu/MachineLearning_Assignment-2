# Breast Cancer Classification (UCI Dataset)
 
## Problem Statement
The goal of this project is to classify tumors as malignant (M) or benign (B) using multiple machine learning models. The assignment demonstrates end-to-end ML workflow: dataset preparation, model training, evaluation, and deployment via a Streamlit app.
 
## Dataset Description
- Source: UCI ML Repository – Breast Cancer Wisconsin (Diagnostic)
- Instances: 569
- Features: 30 real-valued features computed from digitized images of fine needle aspirates of breast masses
- Target: Malignant (M=1) vs Benign (B=0)
 
## Models Used
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost
 
## Comparison Table (Evaluation Metrics)
| Model               |   Accuracy |      AUC |   Precision |   Recall |       F1 |      MCC |
|:--------------------|-----------:|---------:|------------:|---------:|---------:|---------:|
| Logistic Regression |   0.982456 | 0.998089 |    0.96875  | 0.984127 | 0.976378 | 0.962501 |
| Decision Tree       |   0.947368 | 0.95172  |    0.897059 | 0.968254 | 0.931298 | 0.890447 |
| KNN                 |   0.959064 | 0.97766  |    0.951613 | 0.936508 | 0.944    | 0.911818 |
| Naive Bayes         |   0.935673 | 0.992651 |    0.919355 | 0.904762 | 0.912    | 0.861382 |
| Random Forest       |   0.976608 | 0.997281 |    0.983607 | 0.952381 | 0.967742 | 0.949705 |
| XGBoost             |   0.964912 | 0.995003 |    0.938462 | 0.968254 | 0.953125 | 0.925387 |


## Observations on Model Performance
- Logistic Regression: Strong baseline with high accuracy and excellent AUC.
- Decision Tree: Good accuracy but slightly lower AUC, shows signs of overfitting.
- KNN: Performance drops significantly, especially recall.
- Naive Bayes: Very poor precision and recall, independence assumptions don’t hold.
- Random Forest: Excellent performance, robust and generalizes well.
- XGBoost: Best overall metrics with highest AUC, balances precision and recall effectively.
 
## Streamlit App Features
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix or classification report

## Repository Structure
project-folder/
├── app.py
├── requirements.txt
├── README.md
└── model/
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── KNN.pkl
    ├── Naive_Bayes.pkl
    ├── Random_Forest.pkl
    └── XGBoost.pkl
