import pandas as pd

# Load the CSV
df = pd.read_csv('data/processed/cleaned_diabetes_data.csv')

print(f'Original columns: {len(df.columns)}')
print(f'Diagnosis column present: {"Diagnosis" in df.columns}')

# Remove Diagnosis column if it exists
if 'Diagnosis' in df.columns:
    df = df.drop('Diagnosis', axis=1)
    print(f'Removed Diagnosis column')
    
print(f'New columns: {len(df.columns)}')

# Save back
df.to_csv('data/processed/cleaned_diabetes_data.csv', index=False)
print('✅ Saved updated CSV file without Diagnosis column')
