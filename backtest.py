import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, datetime, timedelta
import warnings

warnings.simplefilter('ignore')

directory = '.\data'
df_dict = {}

#'''
for f in os.listdir(directory):
    if f.endswith('.csv'):
        raw_df = (pd.read_csv(f'{directory}\{f}', skipinitialspace=True))
        cols = ['[QUOTE_DATE]','[UNDERLYING_LAST]','[EXPIRE_DATE]','[DTE]','[C_BID]','[C_ASK]','[STRIKE]','[P_BID]','[P_ASK]','[STRIKE_DISTANCE]','[STRIKE_DISTANCE_PCT]']
        raw_df[cols] = raw_df[cols].fillna(0) #TO CHANGE IF IT CAUSES CALC ERRORS
        df = raw_df[cols]
        df_dict[f'{os.path.splitext(f)[0][-6:]}'] = df
'''
#FOR TESTING
n=0
for f in os.listdir(directory):
    if f.endswith('.csv'):
        raw_df = (pd.read_csv(f'{directory}\{f}', skipinitialspace=True))
        cols = ['[QUOTE_DATE]','[UNDERLYING_LAST]','[EXPIRE_DATE]','[DTE]','[C_BID]','[C_ASK]','[STRIKE]','[P_BID]','[P_ASK]','[STRIKE_DISTANCE]','[STRIKE_DISTANCE_PCT]']
        raw_df[cols] = raw_df[cols].fillna(0) #TO CHANGE IF IT CAUSES CALC ERRORS
        df = raw_df[cols]
        df_dict[f'{os.path.splitext(f)[0][-6:]}'] = df
    n+=1
    if n==6:
        break
#'''
print('dfs loaded')

#vars
rf = 0.01
#auxiliary functions
def adddate(d1: str, diff: int):
    d2 = date.fromisoformat(d1)+timedelta(diff)
    return d2.strftime('%Y-%m-%d')

def datediff(start: str, end: str):
    k = date.fromisoformat(end) - date.fromisoformat(start)
    return k.days

def sharpe(ts, rf = rf, N = 255):
    pct = ts.pct_change(1)
    mu = pct.mean() * N -rf
    sigma = pct.std() * np.sqrt(N)
    return round(mu/sigma,5)

def sortino(ts, rf = rf, N = 255):
    pct = ts.pct_change(1)
    mu = pct.mean() * N -rf
    sigma = pct[pct<0].std()*np.sqrt(N)
    return round(mu/sigma,5)

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
            instrument.loc['CURRENT_PRICE'] = currentprice.item()
        instrument.loc['PNL'] = (instrument['QUANTITY']*(instrument['CURRENT_PRICE']))
        return instrument
    def updatepnl(self,day,daychain):
        self.portfolio = self.portfolio.apply(lambda row: self.calcpnl(row, day, daychain), axis=1)
        if self.portfolio['INSTRUMENT'].tolist() != []:
            cond = self.portfolio['INSTRUMENT'].str.split(',').str[2].apply(lambda row: datediff(day,row)<=0)
            i = self.portfolio[cond].index
            if len(i.tolist()) != 0:
                pi = self.portfolio.loc[i,'CURRENT_PRICE'].dot(self.portfolio.loc[i,'QUANTITY']).item()
                self.cash += pi
                self.unrealized -= pi
            self.portfolio = self.portfolio.drop(i)
        if self.portfolio['PNL'].tolist() != []:
            self.unrealized = round(self.portfolio['PNL'].sum(),2)

    def trade(self, instrument, quantity, price):
        self.cash -= quantity*price
        self.unrealized += quantity*price
        i = self.portfolio.index[self.portfolio['INSTRUMENT'] == instrument].tolist()
        if len(i) > 1:
            raise ValueError('SUM TING WONG')
        elif len(i) == 1:
            q = self.portfolio.loc[i[0],'QUANTITY'].item() #need to change df directly
            p = self.portfolio.loc[i[0],'AVG_PRICE'].item()
            if q + quantity != 0:
                self.portfolio.loc[i[0],'AVG_PRICE'] = (q*p+quantity*price)/(q+quantity)
                self.portfolio.loc[i[0],'QUANTITY'] += quantity
            else:
                self.portfolio = self.portfolio.drop(i)
        else:
            row = {'INSTRUMENT':instrument,
                   'QUANTITY':quantity,
                   'AVG_PRICE':price,
                   'CURRENT_PRICE':price,
                   'PNL':quantity*price}
            self.portfolio = pd.concat([self.portfolio,pd.DataFrame.from_dict([row])],ignore_index=True)

class Strategy:
    def __init__(self, strat='put', rollfreq='1m', strikedistance=0.2, pctcash=0.2, buydte=28, ratio=1):
        strats = ['put', 'straddle','strangle']
        rollfreqs = {'1w':7,'2w':14,'3w':21,'1m':28,'2m':56,'3m':84}
        self.strat = strat
        if self.strat not in strats:
            raise ValueError(f'supported strats include {strats}')
        self.rollfreq = rollfreqs[rollfreq]
        self.sd = strikedistance
        self.pctcash = pctcash #TODO KELLY CRITERION
        self.buydte = buydte
        self.ratio = ratio

    def findprice(self,df,inst):
        item = df[(df['[STRIKE]'] == inst[0]) & (df['[DTE]'] == inst[2])]
        '''
        dict = {('call','buy'):item['[C_ASK]'].item(),
                ('call','sell'):item['[C_BID]'].item(),
                ('put','buy'):item['[P_ASK]'].item(),
                ('put','sell'):item['[P_BID]'].item()}
        return dict[(inst[1],inst[3])] ''' #TODO: weighted midprice? change inst[3] to sign(inst)
        dict = {'call': (item['[C_ASK]'].item()+item['[C_BID]'].item())/2,
                'put': (item['[P_ASK]'].item()+item['[P_BID]'].item())/2}
        return dict[inst[1]]
    def closeststrike(self,daychain,dte,sd,moneyness='itm'): #CURRENTLY DOES NOT SUPPORT ATM
        daychain['diff'] = abs(daychain['[STRIKE_DISTANCE_PCT]'] - sd)
        underlying = daychain['[UNDERLYING_LAST]'].iloc[0]
        if moneyness == 'itm':
            filtered = daychain[(daychain['[STRIKE]']>underlying) & (daychain['[DTE]'] == dte)]
        elif moneyness == 'otm':
            filtered = daychain[(daychain['[STRIKE]']<underlying) & (daychain['[DTE]'] == dte)]
        i = filtered['diff'].idxmin()
        return daychain.loc[i,'[STRIKE]']
    def closestdte(self,daychain,dte):
        d = daychain['[DTE]'].unique()
        return int(min(d, key=lambda x:abs(x-dte)))

    def buyrules(self,daychain):
        #this is where the rules are defined
        if self.strat == 'put':
            dte = self.closestdte(daychain, self.buydte)
            sp = self.closeststrike(daychain, dte, self.sd,'itm')
            t = self.findprice(daychain, [sp, 'put', dte, 1])
            qty = round(self.portfolio.cash * self.pctcash / (100 * t)) * 100
            orders = [[sp, 'put', dte, qty]]
        elif self.strat == 'straddle':
            dte = self.closestdte(daychain, self.buydte)
            sp = self.closeststrike(daychain, dte, self.sd,'itm')
            p = self.findprice(daychain, [sp, 'put', dte, 1])
            c = self.findprice(daychain, [sp, 'call', dte, 1])
            t = self.ratio*p+c
            base = self.portfolio.cash * self.pctcash/t
            cqty = round(base/100)*100
            pqty = round(self.ratio*base/100)*100
            orders = [[sp, 'put', dte, pqty], [sp, 'call', dte, cqty]]
        elif self.strat == 'strangle':
            dte = self.closestdte(daychain, self.buydte)
            psp = self.closeststrike(daychain, dte, self.sd,'itm')
            p = self.findprice(daychain, [psp, 'put', dte, 1])
            csp = self.closeststrike(daychain, dte, self.sd*1.5, 'itm')
            c = self.findprice(daychain, [csp, 'call', dte, 1])
            t = self.ratio*p + c
            base = self.portfolio.cash * self.pctcash / t
            cqty = round(base / 100) * 100
            pqty = round(self.ratio * base / 100) * 100
            orders = [[psp, 'put', dte, pqty], [csp, 'call', dte, cqty]]
        return orders

    def backtest(self):
        results = []
        startdate = date.fromisoformat(df_dict['201601']['[QUOTE_DATE]'][0])
        #enddate = date.fromisoformat(df_dict['201605']['[QUOTE_DATE]'].iat[-1])
        enddate = date.fromisoformat(df_dict['202312']['[QUOTE_DATE]'].iat[-1])
        if self.rollfreq == 7:
            exp_1w_raw = pd.date_range(startdate, enddate, freq='W-FRI')
            self.exp = pd.DataFrame()
            self.exp['date'] = exp_1w_raw.date
            self.exp['yearmonth'] = self.exp['date'].apply(lambda x: x.strftime('%Y%m'))
        elif self.rollfreq == 28:
            exp_1m_raw = pd.date_range(startdate, enddate, freq='WOM-3FRI')
            self.exp = pd.DataFrame()
            self.exp['date'] = exp_1m_raw.date
            self.exp['yearmonth'] = self.exp['date'].apply(lambda x: x.strftime('%Y%m'))

        self.portfolio = Portfolio()
        self.buydates = self.exp['date'].tolist()
        self.selldates = self.exp['date'].tolist()[1:]
        wtfdates = ['2018-09-21','2018-12-21','2019-03-01','2019-05-03','2019-05-17','2019-05-24','2019-06-07']
        #bro how the fuck does an option just disappear into thin air man

        for month in self.exp['yearmonth'].unique().tolist():
            df = df_dict[month]
            for day in df['[QUOTE_DATE]'].unique().tolist():
                daychain = df[df['[QUOTE_DATE]'] == day]
                if date.fromisoformat(day) in self.selldates:
                    for index, row in self.portfolio.portfolio.iterrows():
                        ins = row['INSTRUMENT']
                        t = ins.split(',')
                        order = [float(t[0]),t[1],datediff(day,t[2]),'sell']
                        qty = -row['QUANTITY']
                        if day not in wtfdates:
                            self.portfolio.trade(ins,qty,self.findprice(daychain, order))
                if date.fromisoformat(day) in self.buydates:
                    orders = self.buyrules(daychain)
                    for order in orders: #[sp, type, dte, qty]
                        p = self.findprice(daychain,order)
                        self.portfolio.trade(f'{order[0]},{order[1]},{adddate(day,order[2])}',
                                         order[3],p)
                self.portfolio.updatepnl(day,daychain)
                results.append(pd.DataFrame({'date':[day], 'pnl':[self.portfolio.cash + self.portfolio.unrealized]}))
        self.resultdf = pd.concat(results)

#main loop
std=0.5
for s in [['put',1,'1m',0.2],['strangle',5,'1m',0.2]]:
    strat = s[0]
    ratio = s[1]
    rollfreq = s[2]
    pctcash = s[3]
    strategy1 = Strategy(strikedistance=std,pctcash=pctcash,rollfreq=rollfreq,strat=strat,ratio=ratio)
    strategy1.backtest()
    pnl1 = strategy1.resultdf['pnl']
    sharpe1 = sharpe(pnl1)
    sortino1 = sortino(pnl1)
    cagr1 = round((pnl1.iloc[-1]/pnl1.iloc[0])**(1/8)-1,5)
    plt.plot(strategy1.resultdf['date'],pnl1,label=f'itm:{std} cash%=0.2 {strat} p/c:{ratio} freq={rollfreq}/1m sh:{sharpe1} so:{sortino1} cagr={cagr1}')
'''
strategy2 = Strategy(strikedistance=std,pctcash=0.2,rollfreq='1w',buydte=7)
strategy2.backtest()
pnl2 = strategy2.resultdf['pnl']
sharpe2 = sharpe(pnl2)
sortino2 = sortino(pnl2)
cagr2 = round((pnl2.iloc[-1]/pnl2.iloc[0])**(1/8)-1,5)
plt.plot(strategy2.resultdf['date'],pnl2,label=f'itm%={std} cash%=0.2 freq=1w trade=1w sharpe={sharpe2} sortino={sortino2} cagr={cagr2}')

strategy3 = Strategy(strikedistance=std,pctcash=0.1,rollfreq='1w')
strategy3.backtest()
pnl3 = strategy3.resultdf['pnl']
sharpe3 = sharpe(pnl3)
sortino3 = sortino(pnl3)
cagr3 = round((pnl3.iloc[-1]/pnl3.iloc[0])**(1/8)-1,5)
plt.plot(strategy3.resultdf['date'],pnl3,label=f'itm%={std} cash%=0.1 freq=1w trade=1m sharpe={sharpe3} sortino={sortino3} cagr={cagr3}')
'''

plt.xticks(['2016-01-04','2017-01-03','2018-01-02','2019-01-02','2020-01-02','2021-01-04','2022-01-03','2023-01-03'])
plt.legend()
plt.show()
print('done')