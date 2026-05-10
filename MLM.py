from ast import Load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Cleaned Data
print("Loading Cleaned Data...\n")
df_age = pd.read_csv('jadual_a1_umur_jumlah_cleaned.csv').dropna(subset=['Mean_Salary'])
df_edu = pd.read_csv('jadual_a3_tp_jum_cleaned.csv').dropna(subset=['Mean_Salary'])

# Descriptive Statistics
print("--- DESCRIPTIVE STATISTICS (Education Dataset) ---")
mean_sal = df_edu['Mean_Salary'].mean()
median_sal = df_edu['Mean_Salary'].median()
sd_sal = df_edu['Mean_Salary'].std()

# Calculate Interquartile Range (IQR)
q3, q1 = np.percentile(df_edu['Mean_Salary'], [75, 25])
iqr_sal = q3 - q1

print(f"Mean Salary: RM {mean_sal:.2f}")
print(f"Median Salary: RM {median_sal:.2f}")
print(f"Standard Deviation: RM {sd_sal:.2f}")
print(f"Interquartile Range (IQR): RM {iqr_sal:.2f}\n")

# Statistical Inference
print("--- HYPOTHESIS TEST 1: EDUCATION (Tertiary vs Secondary) ---")
salaries_tertiary = df_edu[df_edu['Category'] == 'Tertiary']['Mean_Salary']
salaries_secondary = df_edu[df_edu['Category'] == 'Secondary']['Mean_Salary']

t_stat_edu, p_val_edu = stats.ttest_ind(salaries_tertiary, salaries_secondary, equal_var=False)
print(f"T-statistic: {t_stat_edu:.4f}, P-value: {p_val_edu:.4e}\n")

print("--- HYPOTHESIS TEST 2: AGE GROUP (Midpoint 27.0 vs 47.0) ---")
salaries_young = df_age[df_age['Age_Midpoint'] == 27.0]['Mean_Salary']
salaries_senior = df_age[df_age['Age_Midpoint'] == 47.0]['Mean_Salary']

t_stat_age, p_val_age = stats.ttest_ind(salaries_young, salaries_senior, equal_var=False)
print(f"T-statistic: {t_stat_age:.4f}, P-value: {p_val_age:.4e}\n")

# Machine Learning Classification
print("--- MACHINE LEARNING: RANDOM FOREST CLASSIFICATION ---")

df_ml = df_age[['Year', 'Age_Midpoint', 'Mean_Salary']].copy()
df_ml['Salary_Bracket'] = pd.qcut(df_ml['Mean_Salary'], q=3, labels=['Low', 'Medium', 'High'])

X = df_ml[['Year', 'Age_Midpoint']]
y = df_ml['Salary_Bracket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
labels = ['High', 'Medium', 'Low']
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Random Forest Confusion Matrix: Predicting Salary Brackets')
plt.ylabel('Actual Salary Bracket')
plt.xlabel('Predicted Salary Bracket')
plt.tight_layout()

plt.savefig('ml_confusion_matrix.png', dpi=300)
plt.close()

print("\nSuccess: Saved visual Confusion Matrix as 'ml_confusion_matrix.png'")