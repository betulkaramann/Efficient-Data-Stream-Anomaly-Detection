# Efficient-Data-Stream-Anomaly-Detection
# Anomaly Detection Comparison for Credit Card Fraud

This project compares three different anomaly detection algorithms for detecting credit card fraud. The algorithms evaluated are:

- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **One-Class SVM**

## Overview

The code in this project performs the following tasks:
1. Loads a credit card fraud dataset.
2. Samples 1,000 records from the dataset.
3. Separates normal transactions from fraud cases.
4. Creates synthetic outlier data.
5. Applies each anomaly detection algorithm to the combined dataset.
6. Evaluates and prints the performance metrics of each algorithm, including accuracy and classification reports.

## Requirements

To run this project, you need the following Python packages:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

This project uses the credit card fraud dataset available from Kaggle.

![image](https://github.com/user-attachments/assets/6b96f147-809f-480a-a2c5-13042d026134)

