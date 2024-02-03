import os
import pandas as pd

directory = '.'
for f in os.listdir(directory):
    if f.endswith('.csv'):
        df = pd.read_csv(f'{directory}\{f}', skipinitialspace=True)
        df['[DTE]'] = df['[DTE]'].apply(round)
        df.to_csv(f'{f}')
