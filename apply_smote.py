from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the class distribution (labels 0, 1, 2)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print("Original class distribution:", dict(pd.Series(y).value_counts()))
print("Balanced class distribution:", dict(pd.Series(y_smote).value_counts()))
