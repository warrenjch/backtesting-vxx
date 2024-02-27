import os
import pandas as pd

directory = '.\\data_vix'
'''
for r, d, f in os.walk(directory,topdown=False):
    for n in f:
        p = f'{r}\{n}'
        df = pd.read_csv(p,sep=',',skipinitialspace=True)
        df.to_csv(f'{directory}\{n[:-4]}.csv')
'''
for f in os.listdir(directory):
    if f.endswith('.csv'):
        df = pd.read_csv(f'{directory}\{f}', skipinitialspace=True)
        df['[DTE]'] = df['[DTE]'].apply(round)
        df.to_csv(f'{directory}\{f}')
