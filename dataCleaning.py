import numpy as np
import pandas as pd

file_path = "salaries_wages_2024.xlsx"
sheet_conf = {"JADUAL A1 Umur Jumlah": "Age group",
                  "JADUAL A3 TP Jum": "Educational attainment"}
df_dict = pd.read_excel(file_path, skiprows=3, sheet_name=list(sheet_conf.keys()))

cleaned_data = {}

for sheet_name, cat_col_name in sheet_conf.items():
    print(f"Cleaning {sheet_name}...")

    # Grab the raw data for this specific sheet 
    df_raw = df_dict[sheet_name]

    # Find the year columns dynamically
    years = [col for col in df_raw.columns if str(col).isdigit() and 2010 <= int(col) <= 2024]
    cols_to_keep = [cat_col_name] + years

    # Filter and rename the category column to a standard name: Category
    df = df_raw[cols_to_keep].copy()
    df.rename(columns={cat_col_name: "Category"}, inplace=True)

    # Breakpoints finding
    rec_idx = df[df["Category"].str.contains("Recipients", na=False)].index[0]
    med_idx = df[df["Category"].str.contains("Median", na=False)].index[0]
    mean_idx = df[df["Category"].str.contains("Mean", na=False)].index[0]

    # Table slicing
    df_recipients = df.iloc[rec_idx + 1 : med_idx].copy()
    df_median = df.iloc[med_idx + 1 : mean_idx].copy()
    df_mean = df.iloc[mean_idx + 1 :].copy()

    # Melt Wide to Long
    melt_rec = df_recipients.melt(id_vars="Category", var_name="Year", value_name="Recipients_000")
    melt_med = df_median.melt(id_vars="Category", var_name="Year", value_name="Median_Salary")
    melt_mean = df_mean.melt(id_vars="Category", var_name="Year", value_name="Mean_Salary")

    # Merge them together
    clean_df = pd.merge(melt_rec, melt_med, on=["Category", "Year"])
    clean_df = pd.merge(clean_df, melt_mean, on=["Category", "Year"])
    
    # Remove "Total" rows and force numbers
    clean_df = clean_df.dropna(subset=["Category"])
    clean_df = clean_df[~clean_df["Category"].str.contains("Total", case=False)]
    
    clean_df["Year"] = pd.to_numeric(clean_df["Year"])
    clean_df["Recipients_000"] = pd.to_numeric(clean_df["Recipients_000"], errors="coerce")
    clean_df["Median_Salary"] = pd.to_numeric(clean_df["Median_Salary"], errors="coerce")
    clean_df["Mean_Salary"] = pd.to_numeric(clean_df["Mean_Salary"], errors="coerce")
    
    clean_df = clean_df.dropna()
    
    # Only calculate Midpoint if we are currently looking at the Age sheet
    if "Umur" in sheet_name:
        def calculate_midpoint(age_string):
            if "-" in age_string:
                parts = age_string.split("-")
                return (int(parts[0]) + int(parts[1])) / 2
            elif "65" in age_string:
                return 67.5 
            return None
        clean_df["Age_Midpoint"] = clean_df["Category"].apply(calculate_midpoint)
        clean_df = clean_df.sort_values(by=["Age_Midpoint", "Year"]).reset_index(drop=True)
    else:
        # For Education, just sort by Year and Category
        clean_df = clean_df.sort_values(by=["Year", "Category"]).reset_index(drop=True)
    
    # Store the cleaned dataframe into new dictionary
    cleaned_data[sheet_name] = clean_df

print("\nAll sheets cleaned successfully!")

# Save each cleaned DataFrame to a separate CSV
for sheet_name, df_final in cleaned_data.items():
    # Clean up the filename (remove spaces and special characters)
    file_name = sheet_name.replace(" ", "_").lower() + "_cleaned.csv"
    
    # Export to CSV (index=False prevents an extra row-number column)
    df_final.to_csv(file_name, index=False)
    
    print(f"Successfully saved: {file_name}")

# fully cleaned, regression-ready data:
df_clean_age = cleaned_data["JADUAL A1 Umur Jumlah"]
df_clean_edu = cleaned_data["JADUAL A3 TP Jum"]

print("\n--- SNEAK PEEK: AGE DATA ---")
print(df_clean_age)

print("\n--- SNEAK PEEK: EDUCATION DATA ---")
print(df_clean_edu)
