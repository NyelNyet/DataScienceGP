import numpy as np
import pandas as pd

file_path = "salaries_wages_2024.xlsx"
fp_sheet_names = ["JADUAL A1 Umur Jumlah",
                  "JADUAL A3 TP Jum"]
df = pd.read_excel(file_path, skiprows=3, sheet_name=fp_sheet_names)

print("\n------AGE GROUP------")
print(df[fp_sheet_names[0]].head())

print("\n------EDUCATION GROUP------")
print(df[fp_sheet_names[1]].head())

