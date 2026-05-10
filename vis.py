import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Load the Cleaned Data
# ==========================================
print("Loading Data for Comprehensive EDA...")
df_clean_age = pd.read_csv('jadual_a1_umur_jumlah_cleaned.csv')
df_clean_edu = pd.read_csv('jadual_a3_tp_jum_cleaned.csv')

sns.set_theme(style="whitegrid")

# ==========================================
# Plot 1: Boxplot for Education (Outliers & Spread)
# ==========================================
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean_edu, x='Category', y='Mean_Salary')
plt.title('Salary Distribution by Education Level')
plt.ylabel('Mean Monthly Salary (RM)')
plt.xlabel('Education Level')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('eda_boxplot_education.png', dpi=300)
plt.close()
print("Saved 1/4: Education Boxplot")

# ==========================================
# Plot 2: Boxplot for Age Brackets (Outliers & Spread)
# ==========================================
plt.figure(figsize=(12, 6))
# Sort so the age brackets appear in chronological order
df_clean_age_sorted = df_clean_age.sort_values('Age_Midpoint')
sns.boxplot(data=df_clean_age_sorted, x='Category', y='Mean_Salary')
plt.title('Salary Distribution by Age Group')
plt.ylabel('Mean Monthly Salary (RM)')
plt.xlabel('Age Group')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('eda_boxplot_age.png', dpi=300)
plt.close()
print("Saved 2/4: Age Boxplot")

# ==========================================
# Plot 3: Histogram (Skewness Check)
# ==========================================
plt.figure(figsize=(10, 6))
# A histogram with a KDE (Kernel Density Estimate) line to clearly show the skew
sns.histplot(df_clean_age['Mean_Salary'].dropna(), kde=True, bins=20, color='royalblue')
plt.title('Distribution of Mean Monthly Salaries (Checking for Right-Skew)')
plt.xlabel('Mean Monthly Salary (RM)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda_histogram_skewness.png', dpi=300)
plt.close()
print("Saved 3/4: Salary Histogram")

# ==========================================
# Plot 4: Correlation Matrix (Relationships)
# ==========================================
plt.figure(figsize=(8, 6))
# Isolate numeric columns to see how they interact
corr_cols = ['Year', 'Age_Midpoint', 'Recipients_000', 'Median_Salary', 'Mean_Salary']
corr_matrix = df_clean_age[corr_cols].corr()

# Use a heatmap to make the numbers easy to read
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix (Age Dataset)')
plt.tight_layout()
plt.savefig('eda_correlation_matrix.png', dpi=300)
plt.close()
print("Saved 4/4: Correlation Matrix Heatmap")