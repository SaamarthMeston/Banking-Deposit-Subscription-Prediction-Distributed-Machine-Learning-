# 🏦 Banking Term Deposit Subscription Prediction using Distributed Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.1-orange?logo=apache-spark)](https://spark.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](Banking_Term_Deposit_Subscription_Prediction_using_Distributed_Machine_Learning.ipynb)

> A distributed machine learning pipeline to predict whether a customer will subscribe to a bank term deposit — built on Apache Spark (PySpark), Spark ML, Spark SQL (Hive-style), and Spark Streaming.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Key Results](#key-results)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [EDA Highlights](#eda-highlights)
- [ML Models](#ml-models)
- [Spark Streaming](#spark-streaming)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

---

## 📖 Project Overview

This project demonstrates a **complete distributed machine learning pipeline** applied to a real-world banking dataset. Using **Apache Spark** as the distributed computing backbone, it covers:

- Distributed data ingestion and Hive-style SQL querying
- Exploratory Data Analysis (EDA) with Spark + Pandas/Matplotlib
- Feature engineering and preprocessing via Spark ML Pipelines
- Training and comparing 3 ML classification models
- Real-time transaction monitoring with Spark Streaming
- Model saving and inference on unseen customer data

---

## 🎯 Problem Statement

A Portuguese banking institution conducts telemarketing campaigns to promote **term deposit subscriptions**. Contacting every customer is inefficient and costly. This project builds a **predictive model** that identifies customers most likely to subscribe — enabling targeted, ROI-positive outreach.

**Challenge:** The dataset represents a sample of what would be millions of records in production. Single-node ML cannot scale. This project solves the problem using **distributed computing with Apache Spark**.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Distributed Computing | Apache Spark (PySpark 3.5.1) |
| Distributed Storage | HDFS simulation via Spark |
| SQL Querying | Spark SQL (Hive-style) |
| Machine Learning | Spark MLlib / Spark ML Pipelines |
| Real-Time Processing | Spark Streaming |
| Visualization | Matplotlib, Seaborn |
| Data Manipulation | Pandas, NumPy |
| Model Persistence | Joblib / Pickle |
| Environment | Google Colab (A100 GPU), OpenJDK 17 |

---

## 📊 Dataset

**Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

| Property | Value |
|---|---|
| Records | 4,521 |
| Features | 16 (+ 1 target) |
| Target Variable | `y` — Term deposit subscribed (yes/no) |
| Class Balance | ~88.5% No / ~11.5% Yes (imbalanced) |
| Missing Values | None |

### Feature Summary

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Customer age (19–87) |
| `job` | Categorical | Job type (12 categories) |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `default` | Categorical | Has credit in default? |
| `balance` | Numerical | Average yearly balance (euros) |
| `housing` | Categorical | Has housing loan? |
| `loan` | Categorical | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `duration` | Numerical | Last call duration (seconds) |
| `campaign` | Numerical | Number of contacts this campaign |
| `pdays` | Numerical | Days since last campaign contact |
| `previous` | Numerical | Number of previous contacts |
| `poutcome` | Categorical | Previous campaign outcome |
| `y` | Target | Subscribed to term deposit? |

---

## 🏗️ Project Architecture

```
Raw Data (bank.csv)
        │
        ▼
┌─────────────────────────────┐
│  HDFS Simulation (Spark)    │  ← Distributed data ingestion
│  Hive SQL via Spark SQL     │  ← Business queries
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   EDA & Hypothesis Testing  │  ← Spark + Pandas + Seaborn
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Feature Engineering        │  ← Wrangling, encoding, scaling
│  Spark ML Pipeline          │  ← StringIndexer → OHE → VectorAssembler → StandardScaler
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  ML Model Training          │
│  ├── Logistic Regression    │
│  ├── Decision Tree          │
│  └── Random Forest ★ Best  │
│  + 3-Fold Cross Validation  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Spark Streaming            │  ← Real-time transaction simulation
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Model Persistence          │  ← Joblib export + inference on unseen data
└─────────────────────────────┘
```

---

## 📈 Key Results

### Model Comparison

| Model | AUC-ROC | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | ~0.88 | ~0.87 | ~0.72 | ~0.68 | ~0.77 |
| Decision Tree | ~0.84 | ~0.85 | ~0.70 | ~0.65 | ~0.76 |
| **Random Forest ★** | **~0.91** | **~0.89** | **~0.76** | **~0.74** | **~0.79** |

> **Random Forest** is the best-performing model, offering the highest AUC-ROC and F1-Score with robust handling of class imbalance via sample weighting.

### Top Predictive Features

1. `duration` — Call length is the strongest predictor (r ≈ 0.40 with target)
2. `poutcome` — Previous campaign success → ~65% subscription rate
3. `job` — Retired (~25%) and student (~28%) segments outperform average (11.5%)
4. `balance` — Higher balance correlates positively with subscription
5. `age_group` — U-shaped relationship: seniors and young customers convert better

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- Java (OpenJDK 17 recommended)
- Google Colab (recommended) or local Jupyter environment

### Install Dependencies

```bash
pip install pyspark==3.5.1 findspark pandas numpy matplotlib seaborn scikit-learn joblib
```

### Java Setup (if running locally)

```bash
# Ubuntu/Debian
sudo apt-get install openjdk-17-jdk-headless

# macOS (Homebrew)
brew install openjdk@17
```

### Dataset

Download `bank.csv` from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and place it in the project root directory.

---

## 🚀 Usage

### 1. Clone the Repository

```bash
git clone https://github.com/umerulla/Banking-Term-Deposit-Subscription-Prediction-using-Distributed-Machine-Learning.git
cd Banking-Term-Deposit-Subscription-Prediction-using-Distributed-Machine-Learning
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook Banking_Term_Deposit_Subscription_Prediction_using_Distributed_Machine_Learning.ipynb
```

### 3. Running on Google Colab

1. Upload the notebook to [Google Colab](https://colab.research.google.com/)
2. Mount your Google Drive and place `bank.csv` at the configured path
3. Run all cells sequentially — Java and PySpark are installed automatically

### 4. Inference on New Customers

After training, the model is saved as `banking_rf_model.joblib`. Load and predict:

```python
import joblib
import pandas as pd

model = joblib.load('banking_rf_model.joblib')

# Example: new customer features
new_customer = pd.DataFrame([{
    'age': 42, 'job': 'management', 'marital': 'married',
    'education': 'tertiary', 'balance': 2500, 'duration': 400,
    'campaign': 1, 'pdays': 0, 'previous': 1
    # ... add all required features
}])

prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)[:, 1]
print(f"Subscription: {'YES ✅' if prediction[0] == 1 else 'NO ❌'} | Probability: {probability[0]:.2%}")
```

---

## 📁 Project Structure

```
├── Banking_Term_Deposit_Subscription_Prediction_using_Distributed_Machine_Learning.ipynb
├── bank.csv                        # Dataset (download separately from UCI)
├── banking_rf_model.joblib         # Saved Random Forest model (generated after training)
├── README.md
└── /tmp/
    ├── bank_stream_input/          # Spark Streaming micro-batch input directory
    └── bank_stream_output/         # Spark Streaming output directory
```

---

## 🔍 EDA Highlights

| Analysis | Key Finding |
|---|---|
| Class Distribution | 88.5% No / 11.5% Yes — heavily imbalanced |
| Call Duration | Subscribers average ~535s vs ~221s for non-subscribers |
| Job Type | Students (28%) & retirees (25%) far above 11.5% average |
| Previous Outcome | `poutcome = success` → ~65% subscription rate |
| Age Pattern | U-shaped: seniors (60+) and youth (<30) convert better |
| Correlation | `duration` (r≈0.40) is the strongest numerical predictor |

### Hypothesis Tests

- **t-test (duration vs subscription):** p < 0.001 → Statistically significant difference ✅
- **Chi-Square (poutcome vs subscription):** p < 0.001 → Strong association ✅

---

## 🤖 ML Models

### 1. Logistic Regression
- Linear baseline model using sigmoid function
- Regularization: `regParam=0.01`, `maxIter=100`
- Class weighting applied for imbalance handling
- Hyperparameter tuning via 3-Fold CrossValidator

### 2. Decision Tree Classifier
- `maxDepth=6`, class-weighted
- Non-linear relationships captured via splits
- Prone to overfitting — controlled via depth and min instances per node

### 3. Random Forest Classifier ★ Best
- Ensemble of 50 trees (`numTrees=50`, `maxDepth=6`)
- Bagging + feature sampling reduces variance
- True parallel training: each tree distributed across Spark executors
- Feature importance extracted from final model

### Imbalance Handling

All models use **class weighting**:
```python
weight_yes = total / (2 * count_yes)   # Higher weight for minority class
weight_no  = total / (2 * count_no)    # Lower weight for majority class
```

---

## ⚡ Spark Streaming

A real-time transaction monitoring pipeline is simulated using Spark Streaming:

- **Micro-batch generation:** Random banking transactions (deposits, withdrawals, transfers, etc.) are written as CSV files to a streaming input directory
- **Streaming analytics:** Transaction type summaries, high-value transaction alerts, and customer segment breakdowns
- **Window operations:** Rolling aggregations partitioned by customer age group using Spark Window functions

```python
# Example: Flag high-value transactions in the stream
stream_df.filter(F.col("amount") > 5000) \
         .select("customer_id", "transaction_type", "amount", "timestamp") \
         .show()
```

---

## 🔮 Future Work

- **Real Kafka integration** — Replace simulated streaming with live Kafka topics
- **MLflow experiment tracking** — Log runs, parameters, and metrics
- **REST API deployment** — Serve the trained model via FastAPI or Flask
- **SHAP explainability** — Explain individual predictions for regulatory compliance
- **Gradient Boosting** — Add GBTClassifier (Spark ML) to the model comparison
- **Delta Lake** — Replace raw CSV ingestion with versioned, ACID-compliant Delta tables
- **Feature Store** — Centralize reusable feature pipelines

---

## 🏁 Conclusion

This project successfully demonstrates a **production-ready distributed ML pipeline** for banking term deposit prediction:

- ✅ Distributed data ingestion and Hive-style SQL analytics via Spark SQL
- ✅ Scalable EDA and statistical hypothesis testing
- ✅ End-to-end Spark ML Pipeline with feature engineering and class imbalance handling
- ✅ Comparison of Logistic Regression, Decision Tree, and Random Forest
- ✅ Hyperparameter tuning via distributed 3-Fold CrossValidator
- ✅ Real-time transaction simulation with Spark Streaming
- ✅ Saved model with inference capability on unseen data

**Random Forest** emerged as the best model with **AUC-ROC ~0.91** and **F1-Score ~0.76**, making it suitable for production deployment in a bank's customer targeting system.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Umer Ulla**  
[GitHub](https://github.com/umerulla) · [LinkedIn](https://linkedin.com/in/umerulla)

---

*If you found this project useful, please consider giving it a ⭐ on GitHub!*
