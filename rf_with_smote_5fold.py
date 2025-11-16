import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# X = df.drop("label", axis=1)
# y = df["label"]

model = RandomForestClassifier(class_weight='balanced', random_state=42)
smote = SMOTE(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_no, acc_sm = [], []
f1_no, f1_sm = [], []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Without SMOTE
    model.fit(X_train, y_train)
    pred_no = model.predict(X_test)
    acc_no.append(accuracy_score(y_test, pred_no))
    f1_no.append(f1_score(y_test, pred_no, average='weighted'))

    # With SMOTE
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model.fit(X_res, y_res)
    pred_sm = model.predict(X_test)
    acc_sm.append(accuracy_score(y_test, pred_sm))
    f1_sm.append(f1_score(y_test, pred_sm, average='weighted'))

# Final results
print("\nRandom Forest Performance (5-Fold CV):")
print(f"Avg Accuracy (No SMOTE): {np.mean(acc_no):.3f}")
print(f"Avg Accuracy (SMOTE):    {np.mean(acc_sm):.3f}")
print(f"Avg F1 Score (No SMOTE): {np.mean(f1_no):.3f}")
print(f"Avg F1 Score (SMOTE):    {np.mean(f1_sm):.3f}")
