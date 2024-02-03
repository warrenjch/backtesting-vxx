import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date, datetime, timedelta
import warnings

warnings.simplefilter('ignore')

directory = '.\data'
df_dict = {}


for f in os.listdir(directory):
    if f.endswith('.csv'):
        raw_df = (pd.read_csv(f'{directory}\{f}', skipinitialspace=True))
        cols = ['[QUOTE_DATE]','[UNDERLYING_LAST]','[EXPIRE_DATE]','[DTE]','[C_BID]','[C_ASK]','[STRIKE]','[P_BID]','[P_ASK]','[STRIKE_DISTANCE]','[STRIKE_DISTANCE_PCT]']
        raw_df[cols] = raw_df[cols].fillna(0) #TO CHANGE IF IT CAUSES CALC ERRORS
        df = raw_df[cols]
        df_dict[f'{os.path.splitext(f)[0][-6:]}'] = df
'''
#FOR TESTING
raw_df = (pd.read_csv('.\data\\uvxy_eod_201602.csv', skipinitialspace=True))
cols = ['[QUOTE_DATE]','[UNDERLYING_LAST]','[EXPIRE_DATE]','[DTE]','[C_BID]','[C_ASK]','[STRIKE]','[P_BID]','[P_ASK]','[STRIKE_DISTANCE]','[STRIKE_DISTANCE_PCT]']
raw_df[cols] = raw_df[cols].fillna(0) #TO CHANGE IF IT CAUSES CALC ERRORS
df = raw_df[cols]
df_dict['201602'] = df
'''
print('dfs loaded')

#auxiliary functions
def adddate(d1: str, diff: int):
    d2 = date.fromisoformat(d1)+timedelta(diff)
    return d2.strftime('%Y-%m-%d')

def datediff(start: str, end: str):
    k = date.fromisoformat(end) - date.fromisoformat(start)
    return k.days

class Portfolio:
    def __init__(self, cash = 10000):
        self.portfolio = pd.DataFrame(columns=['INSTRUMENT','QUANTITY','AVG_PRICE','CURRENT_PRICE','PNL'])
        self.unrealized = 0
        self.cash = cash
        #'instrument' is a TUPLE with two values, STRIKE and CALL/PUT

    def calcpnl(self,instrument,day,daychain):
        inst = instrument['INSTRUMENT'].split(',')
        x = datediff(day,inst[2])
        s = daychain[(daychain['[STRIKE]'] == float(inst[0])) & (daychain['[DTE]'] == x)]
        if inst[1] == 'call':
            currentprice = (s['[C_BID]'] + s['[C_ASK]']) / 2
        elif inst[1] == 'put':
            currentprice = (s['[P_BID]'] + s['[P_ASK]']) / 2
        if len(currentprice) != 0:
            instrument.loc['CURRENT_PRICE'] = currentprice
        instrument.loc['PNL'] = (instrument['QUANTITY']*(instrument['CURRENT_PRICE']-instrument['AVG_PRICE'])).item()
        return instrument
    def updatepnl(self,day,daychain):
        if self.portfolio['INSTRUMENT'].tolist() != []:
            cond = self.portfolio['INSTRUMENT'].str.split(',').str[2].apply(lambda row: datediff(day,row)<0)
            i = self.portfolio[cond].index
            if len(i.tolist()) != 0:
                #print(day, i, '\n', self.portfolio.loc[i, ['INSTRUMENT', 'PNL']]) #TODO REMOVE THIS SHIT
                self.cash += self.portfolio.loc[i,'CURRENT_PRICE'].dot(self.portfolio.loc[i,'QUANTITY']).item()
            self.portfolio = self.portfolio.drop(i)
        self.portfolio = self.portfolio.apply(lambda row: self.calcpnl(row,day,daychain),axis=1)
        if self.portfolio['PNL'].tolist() == []:
            self.unrealized = round(self.cash,2)
        else:
            self.unrealized = round(self.cash + self.portfolio['PNL'].sum(),2)

    def trade(self, instrument, quantity, price):
        self.cash -= quantity*price
        i = self.portfolio.index[self.portfolio['INSTRUMENT'] == instrument].tolist()
        if len(i) > 1:
            raise ValueError('SUM TING WONG')
        elif len(i) == 1:
            q = self.portfolio.loc[i[0],'QUANTITY'].item() #need to change df directly
            p = self.portfolio.loc[i[0],'AVG_PRICE'].item()
            self.portfolio.loc[i[0],'AVG_PRICE'] = (q*p+quantity*price)/(q+quantity)
            self.portfolio.loc[i[0],'QUANTITY'] += quantity
        else:
            row = {'INSTRUMENT':instrument,
                   'QUANTITY':quantity,
                   'AVG_PRICE':price,
                   'CURRENT_PRICE':price,
                   'PNL':0}
            self.portfolio = pd.concat([self.portfolio,pd.DataFrame.from_dict([row])],ignore_index=True)

class Strategy:
    def __init__(self, strat='put', rollfreq='1m', strikedistance=0.2, pctcash=0.2):
        strats = ['put', 'calendar']
        rollfreqs = ['1w','2w','3w','1m','2m','3m']
        self.strat = strat
        self.rollfreq = rollfreq
        self.sd = strikedistance
        self.pctcash = pctcash #TODO KELLY CRITERION
        if self.strat not in strats:
            raise ValueError(f'supported strats include {strats}')
        if self.rollfreq not in rollfreqs:
            raise ValueError(f'supported roll frequencies include {rollfreqs}')

    def findprice(self,df,inst):
        item = df[(df['[STRIKE]'] == inst[0]) & (df['[DTE]'] == inst[2])]
        dict = {('call','buy'):item['[C_ASK]'].item(),
                ('call','sell'):item['[C_BID]'].item(),
                ('put','buy'):item['[P_ASK]'].item(),
                ('put','sell'):item['[P_BID]'].item()}
        return dict[(inst[1],inst[3])]
    def closeststrike(self,daychain,dte): #CURRENTLY ONLY SUPPORTS ITM PUT
        daychain['diff'] = abs(daychain['[STRIKE_DISTANCE_PCT]'] - self.sd)
        underlying = daychain['[UNDERLYING_LAST]'].iloc[0]
        filtered = daychain[(daychain['[STRIKE]']>underlying) & (daychain['[DTE]'] == dte)]
        i = filtered['diff'].idxmin()
        return daychain.loc[i,'[STRIKE]']
    def closestdte(self,daychain,dte):
        d = daychain['[DTE]'].unique()
        return int(min(d, key=lambda x:abs(x-dte)))

    def backtest(self):
        results = []
        startdate = date.fromisoformat(df_dict['201601']['[QUOTE_DATE]'][0])
        #enddate = date.fromisoformat(df_dict['201602']['[QUOTE_DATE]'].iat[-1])
        enddate = date.fromisoformat(df_dict['202312']['[QUOTE_DATE]'].iat[-1])
        exp_1m_raw = pd.date_range(startdate, enddate, freq='WOM-3FRI')
        self.exp_1m = pd.DataFrame()
        self.exp_1m['date'] = exp_1m_raw.date
        self.exp_1m['yearmonth'] = self.exp_1m['date'].apply(lambda x: x.strftime('%Y%m'))
        self.portfolio = Portfolio()
        for index, row in self.exp_1m.iterrows():
            df = df_dict[row['yearmonth']]
            for day in df['[QUOTE_DATE]'].unique().tolist():
                daychain = df[df['[QUOTE_DATE]'] == day]
                if date.fromisoformat(day) == row['date']:
                    dte = self.closestdte(daychain,28)
                    sp = self.closeststrike(daychain,dte)
                    spot = daychain['[UNDERLYING_LAST]'].unique()[0]
                    order = [sp,'put',dte,'buy'] #TODO DATETIME PROGRAMMING, PUT EXPIRY DATE AND CALC
                    self.portfolio.trade(f'{order[0]},{order[1]},{adddate(day,order[2])}',
                                         round(self.portfolio.cash*self.pctcash/spot),
                                         self.findprice(daychain,order))
                self.portfolio.updatepnl(day,daychain)
                results.append(pd.DataFrame({'date':[day], 'pnl':[self.portfolio.unrealized]}))
        self.resultdf = pd.concat(results)
        print(self.resultdf.tail(5))


for i in [0.05,0.3,0.5]:
    strategy = Strategy(strikedistance=i,pctcash=.4)
    strategy.backtest()
    plt.plot(strategy.resultdf['date'],strategy.resultdf['pnl'],label=f'yolo%={i}')

plt.xticks(['2016-01-04','2017-01-03','2018-01-02','2019-01-02','2020-01-02','2021-01-04','2022-01-03','2023-01-03'])
plt.legend()
plt.show()
print('done')