import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your raw data
df_full = pd.read_csv("PNA.csv")

# 2. Add the missing ID column automatically based on the rows
df_full['id'] = df_full.index + 1

# 3. Use the EXACT same random split (random_state=42) as your models file
_, df_validation = train_test_split(df_full, test_size=0.20, random_state=42)

# 4. Filter down to just the columns the human needs to see
df_human_blank = df_validation[['id', 'headlines']].copy()

# 5. Add an empty column where your volunteer will type their answers
df_human_blank['human_label'] = ""

# 6. Save it to your desktop or project folder as an Excel file
output_file = "blind_human_validation_sheet.xlsx"
df_human_blank.to_excel(output_file, index=False)

print(f"🎉 SUCCESS! Clean blank sheet created with {len(df_human_blank)} rows.")
print(f"Go to your folder and find: '{output_file}'")