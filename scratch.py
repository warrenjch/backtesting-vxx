import pandas as pd

# Assuming df is your DataFrame
# Example DataFrame
data = {'instrument': ['A.0.0', 'B.1.0', 'C.0.0', 'D.0.1', 'E.0.0']}
df = pd.DataFrame(data)

# Split the "instrument" column and check the third element
condition = df['instrument'].str.split('.').str[2] == '0'
print(df['instrument'].str.split('.').str[2])

# Filter the DataFrame based on the condition
filtered_df = df[condition]

# Display the result
print(filtered_df)
