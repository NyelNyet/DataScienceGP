import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Load your datasets
df_age = pd.read_csv("jadual_a1_umur_jumlah_cleaned.csv")
df_edu = pd.read_csv("jadual_a3_tp_jum_cleaned.csv")

# PLOT 1: KDE Distribution Plot (Mean Salary)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_age, x="Mean_Salary", fill=True, color="royalblue", alpha=0.5)
plt.title("Distribution of Mean Monthly Salaries (2010 - 2024)", fontsize=14, pad=15)
plt.xlabel("Mean Salary (RM)", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.savefig("1_Salary_Distribution_KDE.png", bbox_inches='tight', dpi=300)
plt.show()

# PLOT 2: Temporal Growth by Age (Line Plot)
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_age, x="Year", y="Mean_Salary", hue="Category", marker="o", palette="viridis")
plt.title("Mean Salary Growth by Age Group Over Time", fontsize=14, pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Mean Salary (RM)", fontsize=12)
plt.legend(title="Age Group", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig("2_Salary_Growth_By_Age.png", bbox_inches='tight', dpi=300)
plt.show()

# PLOT 3: Education Premium (Bar Chart)
# Define the order so it goes from lowest to highest education
edu_order = ["No formal education", "Primary", "Secondary", "Tertiary"]

plt.figure(figsize=(10, 6))
sns.barplot(data=df_edu, x="Category", y="Mean_Salary", order=edu_order, palette="Blues_d")
plt.title("Overall Average Salary by Education Level (2010 - 2024)", fontsize=14, pad=15)
plt.xlabel("Educational Attainment", fontsize=12)
plt.ylabel("Average Mean Salary (RM)", fontsize=12)

plt.savefig("3_Education_Premium_Bar.png", bbox_inches='tight', dpi=300)
plt.show()

# PLOT 4: Education Premium Over Time (Line)
plt.figure(figsize=(12, 7))
sns.lineplot(data=df_edu, x="Year", y="Mean_Salary", hue="Category", hue_order=edu_order, marker="s", palette="Set1")
plt.title("Widening the Gap: Salary Trends by Education Level (2010 - 2024)", fontsize=14, pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Mean Salary (RM)", fontsize=12)
plt.legend(title="Education Level", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig("4_Education_Trends_Line.png", bbox_inches='tight', dpi=300)
plt.show()