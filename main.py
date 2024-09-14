import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
from matplotlib.colors import Normalize
from matplotlib import cm

rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42 
import traceback

# This code is used to compare three different anomaly detection algorithms—
# Isolation Forest, Local Outlier Factor, and One-Class SVM—for detecting credit card fraud.
# It starts by loading the dataset and sampling 1,000 records.
# Then, it separates normal and fraud cases, creates random fake outlier data,
# and applies each algorithm to the combined dataset.
# The code evaluates and prints the performance of each algorithm, including accuracy and classification metrics,
# to see which one performs best at detecting anomalies.



file_path = 'data.csv'

# In this part we try to read the file and handle errors
try:
    df = pd.read_csv(file_path)
    if df is not None and not df.empty:
        print("File loaded successfully.")
    else:
        print("File loaded but Dataframe is empty.")
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc() 
    raise  

# Reduce the dataset to 1000 item, here
df = pd.read_csv(file_path)
df = df.sample(n=1000, random_state=RANDOM_SEED)

invalid = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
# it's calculates the ratio of fraudulent transactions to valid transactions.
outlier_value = len(invalid) / float(len(valid))

columns = df.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]
target = "Class"

state = np.random.RandomState(RANDOM_SEED)
X = df[columns]
Y = df[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))

# The purpose of describing three different outlier detection algorithms is to understand and compare how the various methods detect outliers in your data set.
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X), 
                                        contamination=outlier_value, random_state=RANDOM_SEED, verbose=0),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                               leaf_size=30, metric='minkowski',
                                               p=2, metric_params=None, contamination=outlier_value),
    "Support Vector Machine": OneClassSVM(kernel='rbf', degree=3, gamma=0.1, nu=0.05, 
                                          max_iter=-1)
}

fig, axes = plt.subplots(len(classifiers), 2, figsize=(14, 8 * len(classifiers)))
if len(classifiers) == 1:
    axes = [axes]

#The decision function and classification results are visualized using the SCATTER function.
for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
        # Plot LOF decision function
        axes[i][0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=scores_prediction, cmap='coolwarm', norm=Normalize(vmin=scores_prediction.min(), vmax=scores_prediction.max()))
        axes[i][0].set_title(f"{clf_name} - Decision Function")
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
        scores_prediction = clf.decision_function(X)
        # Plot One-Class SVM decision function
        axes[i][0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=scores_prediction, cmap='coolwarm', norm=Normalize(vmin=scores_prediction.min(), vmax=scores_prediction.max()))
        axes[i][0].set_title(f"{clf_name} - Decision Function")
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
        # Plot Isolation Forest decision function
        axes[i][0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=scores_prediction, cmap='coolwarm', norm=Normalize(vmin=scores_prediction.min(), vmax=scores_prediction.max()))
        axes[i][0].set_title(f"{clf_name} - Decision Function")
    
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    print("{}: {}".format(clf_name, n_errors))
    print("Accuracy Score:")
    print(accuracy_score(Y, y_pred))
    print("Classification Report:")
    print(classification_report(Y, y_pred))
    
    # Plot classification results
    axes[i][1].scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k')
    axes[i][1].set_title(f"{clf_name} - Classification Results")

plt.tight_layout()
plt.show()
