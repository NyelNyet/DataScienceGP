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
print("--- 1. GENERATING DESCRIPTIVE STATISTICS TABLES ---")

# Calculate and Render Education Stats Table
edu_stats = df_edu.groupby('Category')['Mean_Salary'].agg(
    Count='count', Mean='mean', Median='median', Std_Dev='std', Min='min', Max='max'
).round(2).reset_index()
edu_stats.rename(columns={'Category': 'Education Level'}, inplace=True)

fig, ax = plt.subplots(figsize=(12, 4)) 
ax.axis('off')
ax.axis('tight')

table_edu = ax.table(cellText=edu_stats.values.tolist(), colLabels=edu_stats.columns.tolist(), loc='center', cellLoc='center')
table_edu.auto_set_font_size(False)
table_edu.set_fontsize(11)
table_edu.scale(1.2, 2.2)

for (row, col), cell in table_edu.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4c72b0')
    else:
        cell.set_facecolor('#f2f2f2' if row % 2 == 0 else '#ffffff')

plt.title('Descriptive Statistics by Education Level', fontweight='bold', fontsize=14, pad=30)
plt.savefig('eda_education_table_visual.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and Render Overall Distribution Metrics Table
overall_stats = pd.DataFrame({
    'Metric': ['Overall Mean', 'Overall Median', 'Skewness (Right-Skewed)', 'Kurtosis'],
    'Value': [
        round(df_age['Mean_Salary'].mean(), 2),
        round(df_age['Mean_Salary'].median(), 2),
        round(df_age['Mean_Salary'].skew(), 3),
        round(df_age['Mean_Salary'].kurtosis(), 3)
    ]
})

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.axis('off')
table_overall = ax2.table(cellText=overall_stats.values.tolist(), colLabels=overall_stats.columns.tolist(), loc='center', cellLoc='center')
table_overall.auto_set_font_size(False)
table_overall.set_fontsize(11)
table_overall.scale(1.2, 2.2)

for (row, col), cell in table_overall.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#55a868')
    else:
        cell.set_facecolor('#f2f2f2' if row % 2 == 0 else '#ffffff')

plt.title('Overall Salary Distribution Metrics', fontweight='bold', fontsize=14, pad=30)
plt.savefig('eda_overall_stats_visual.png', dpi=300, bbox_inches='tight')
plt.close()
print("Success: Generated visual tables for Descriptive Statistics.\n")

# Statistical Inference
print("--- 2. HYPOTHESIS TESTING (Welch's t-test) ---")
# Education Test: Tertiary vs Secondary
salaries_tertiary = df_edu[df_edu['Category'] == 'Tertiary']['Mean_Salary']
salaries_secondary = df_edu[df_edu['Category'] == 'Secondary']['Mean_Salary']
t_stat_edu, p_val_edu = stats.ttest_ind(salaries_tertiary, salaries_secondary, equal_var=False)
print(f"Education T-test: T-stat = {t_stat_edu:.4f}, P-value = {p_val_edu:.4e}")

# Age Test: Young (27) vs Senior (47)
salaries_young = df_age[df_age['Age_Midpoint'] == 27.0]['Mean_Salary']
salaries_senior = df_age[df_age['Age_Midpoint'] == 47.0]['Mean_Salary']
t_stat_age, p_val_age = stats.ttest_ind(salaries_young, salaries_senior, equal_var=False)
print(f"Age T-test: T-stat = {t_stat_age:.4f}, P-value = {p_val_age:.4e}\n")

# Machine Learning
print("--- 3. MACHINE LEARNING: RANDOM FOREST CLASSIFICATION ---")
df_ml = df_age[['Year', 'Age_Midpoint', 'Mean_Salary']].copy()
df_ml['Salary_Bracket'] = pd.qcut(df_ml['Mean_Salary'], q=3, labels=['Low', 'Medium', 'High'])

X = df_ml[['Year', 'Age_Midpoint']]
y = df_ml['Salary_Bracket']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Render Confusion Matrix Visualization
labels = ['High', 'Medium', 'Low']
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Random Forest Confusion Matrix: Predicting Salary Brackets')
plt.ylabel('Actual Salary Bracket')
plt.xlabel('Predicted Salary Bracket')
plt.tight_layout()
plt.savefig('ml_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSuccess: Generated all terminal outputs and 3 visual PNGs.")