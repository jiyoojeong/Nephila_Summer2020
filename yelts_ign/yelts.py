# coding=utf-8
"""
US renewable pseudo code
- combine all market dfs by adding in the ‘as of date’ and ‘peak type’ as a column
- calculate atc from peak type (this is a row wise addition)
- so at this point you should have 7-8 dfs/sheets with all aggregated data from as of date 2012 to 2020
- using as of date, calculate the day difference between the predicted date (forward date), add this as a column ‘day window’ or something
- group all the ‘day windows’ together - this will be the x axis of analysis
- at this point, get the next month of dates, calculate all statitics using the month(30day) additional to the as of date
- calculate mean, sd, max, min across that x=day window and chart the line plot of the these values for x=day window and y=predicted forward price for on off and atc prices
- we should see an interesting trend that volatility decreases as time increases.

"""

import pandas as pd
import matplotlib
import pprint

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
import os
import re
import datetime
import numpy as np

# ====== GLOBAL VARS ===== #
filedir = '/Users/jiyoojeong/desktop/C/raw/forwards/spg/'


# ====== FUNCTIONS ====== #


def combine_market(market):
    # list all in dir with name == to market

    lis = os.listdir(filedir)
    lis.remove('ERCOT')
    date = datetime.datetime.today()
    yesterday = (date - datetime.timedelta(days=1)).strftime("%m%d%Y")
    d = date.strftime("%m%d%Y")
    if market + '_' + d + '.csv' not in os.listdir(filedir + 'ERCOT'):

        # print(d)
        print('Market: ' + market + '  Day: ' + d)
        big_one = pd.DataFrame()
        count = 0
        for f in lis:
            if re.search(market, f) and not re.search('META', f):
                df = pd.read_csv(filedir + f, engine='python')
                # print(df['Unnamed: 0.1'])

                region = df['Unnamed: 0.1'][2]
                region = region.split(':')[1].strip()
                peak = df['Unnamed: 0.1'][4]
                peak = peak.split(':')[1].strip()
                as_of_date = df['Unnamed: 0.1'][6]  # accounts for weekend dates or days data is not available already
                as_of_date = as_of_date.split(':')[1].strip()
                headers = df.iloc[7, :].values
                headers = np.delete(headers, 0)

                # print(region, peak, as_of_date, headers, type(region), type(headers))

                df = df.drop(df.index[[0, 1, 2, 3, 4, 5, 6, 7]])
                df = df.drop(columns=['Unnamed: 0'])

                # change df to add new categorical parameters
                df.columns = headers
                df['AsOfDate'] = as_of_date
                df['Peak'] = peak
                df['Region'] = region
                # print(df)
                # big_one.columns = headers
                big_one = big_one.append(df)
                # print(big_one)
                count += 1
                if count % 100 == 0:
                    # print(df)
                    print(big_one.tail(10))
        os.chdir(filedir)
        # print(os.listdir(filedir))
        if market not in os.listdir(filedir):
            os.makedirs(market)
        bigfilename = filedir + market + '/' + market + '_' + str(d) + '.csv'
        big_one.to_csv(bigfilename)
        return bigfilename
    return filedir + market + '/' + market + '_' + d + '.csv'


def atc(on, off):
    # print(on[0], off[0])
    # print(type(on[0]), type(off[0]))
    b = (lambda x, y: (16.0 / 24) * x + (8.0 / 24 * 5 / 7 + 24 / 24 * 2 / 7) * y)(on.values, off.values)
    return b


# ====== ERCOT ====== #
ercot_files = os.listdir(filedir + 'ERCOT')
# print(ercot_files)
ercot_file = filedir + 'ERCOT/' + 'ERCOT_07282020.csv'  # combine_market('ERCOT')

ercot = pd.read_csv(ercot_file)
# print(ercot.head())
ercot_cols_float = [u'AEN', u'CPS', u'DC_E', u'DC_N', u'DC_R',
                    u'Houston Zone', u'LCRA', u'North Zone', u'OKLA', u'South Zone',
                    u'West Zone']

for c in ercot_cols_float:
    ercot[c] = ercot[c].astype(float)

ercot['AsOfDate'] = pd.to_datetime(ercot['AsOfDate'])
ercot['Term'] = pd.to_datetime(ercot['Term'])
ercot = ercot.drop(columns=['Unnamed: 0'])
# ercot_early = ercot[ercot['AsOfDate'] <= datetime.datetime.strptime('12/31/2013', '%m/%d/%Y')].dropna()
# print(ercot)


# get atc

zones = [u'Houston Zone', u'North Zone', u'South Zone', u'West Zone']

ercot_on = ercot.loc[ercot['Peak'] == 'On Peak', :].drop(columns=['Peak'])
ercot_off = ercot.loc[ercot['Peak'] == 'Off Peak', :].drop(columns=['Peak'])

# account for incompleteness?
ercot_on_off = ercot_on.merge(right=ercot_off, left_on=['AsOfDate', 'Term'],
                              right_on=['AsOfDate', 'Term'], how='left').assign(
    Houston=lambda df: atc(df['Houston Zone_x'], df['Houston Zone_y']),
    West=lambda df: atc(df['West Zone_x'], df['West Zone_y']),
    South=lambda df: atc(df['South Zone_x'], df['South Zone_y']),
    North=lambda df: atc(df['North Zone_x'], df['North Zone_y']))

ercot_atc = ercot_on_off[['AsOfDate', 'Term', 'Houston', 'West', 'South', 'North']]
ercot_atc = ercot_atc.rename(columns={'Houston': 'Houston Zone', 'West': 'West Zone',
                                      'South': 'South Zone', 'North': 'North Zone'})
ercot_atc['Peak'] = 'ATC'

''' Now we want to get the value difference from the historical to the predicted.'''
''' 
TODO:
- import the historical prices
- match each term date to each historical date
- find the difference
- use this difference as the new values for delta
'''

ERCOT_RT = pd.read_csv('/Users/jiyoojeong/desktop/C/raw/reals/ERCOT/ERCOT_RT.csv')
# ercot_rt_reshaped = ERCOT_RT.pivot(index='date', columns=['Peak', 'Subregion'])
# print(ercot_rt_reshaped)
ercot = ercot[ercot_atc.columns]
ercot = ercot.append(ercot_atc).sort_values(by=['AsOfDate', 'Term', 'Peak']).drop_duplicates()
# ercot = ercot.loc[['Houston Zone', 'North Zone', 'South Zone', 'West Zone']]
# ercot_early = ercot.append(ercot_atc).sort_values(by=['AsOfDate', 'Term', 'Peak'])
# TODO: Some magic and find the differences between prices for each term, peak, and region.

ercot_rt_h = ercot.merge(ERCOT_RT, left_on=['Term', 'Peak'],
                         right_on=['date', 'Peak'], how='left')
print(ercot_rt_h.head())

ercot_delta = ercot.copy()
# ercot_delta_early = ercot_early.copy()
# monthly window averages
# ercot_delta = ercot.sort_values(by=['Term', 'AsOfDate']).rolling(30, min_periods=1).mean()

# day differences from predicted term date and the as of date
ercot_delta['Delta'] = ercot_delta['Term'] - ercot_delta['AsOfDate']
# ercot_delta_early['Delta'] = ercot_delta_early['Term'] - ercot_delta_early['AsOfDate']
# print(ercot_delta.head())
# drop index to make uniform
ercot_delta.reset_index(drop=True)
# melt columns so that the remaining columns are Delta Peak Subregion Value

real_ercot = ercot_delta[ercot_delta['Delta'] >= datetime.timedelta(days=0)]
# real_ercot_early = ercot_delta_early[ercot_delta_early['Delta'] <= datetime.timedelta(days=-20)]
print(real_ercot)

# may also need to look at https://stackoverflow.com/questions/55403008/pandas-partial-melt-or-group-melt
ercot_delta_melt = pd.melt(ercot_delta,
                           id_vars=['Delta', 'Peak'],
                           value_vars=zones,
                           var_name='Subregion',
                           )

ercot_delta_melt = ercot_delta_melt.drop_duplicates()
print(ercot_delta_melt.sort_values(['Delta', 'Subregion', 'Peak'], ascending=True).tail(20))
# group column 'Delta' and find statistics

# mean
ercot_delta_avg = ercot_delta_melt.groupby(['Delta', 'Peak', 'Subregion']).mean()
print(ercot_delta_avg)
# standard deviation
ercot_delta_sd = ercot_delta_melt.groupby(['Delta', 'Peak', 'Subregion']).std()
# max
ercot_delta_max = ercot_delta_melt.groupby(['Delta', 'Peak', 'Subregion']).max()
# min
ercot_delta_min = ercot_delta_melt.groupby(['Delta', 'Peak', 'Subregion']).min()

# os.chdir('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/')
if 'figures' not in os.listdir('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/'):
    os.makedirs('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures')

# plot to see
g = sns.FacetGrid(ercot_delta_avg.reset_index(), col="Subregion", hue='Peak', col_wrap=4)
g.map(plt.plot, 'Delta', 'value', linewidth=.5).add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('ERCOT Means')  # , y=1.05)
# g.fig.title('Ercot Means')
g.fig.savefig('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures/ercot_means.png')

g2 = sns.FacetGrid(ercot_delta_sd.reset_index(), col="Subregion", hue='Peak', col_wrap=4)
g2.map(plt.plot, 'Delta', 'value', linewidth=.5).add_legend()
plt.subplots_adjust(top=0.9)
g2.fig.suptitle('ERCOT Deviation')  # , y=1.05)
g2.fig.savefig('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures/ercot_sd.png')

g3 = sns.FacetGrid(ercot_delta_max.reset_index(), col="Subregion", hue='Peak', col_wrap=4)
g3.map(plt.plot, 'Delta', 'value', linewidth=.5).add_legend()
plt.subplots_adjust(top=0.9)
g3.fig.suptitle('ERCOT Max')  # , y=1.05)
g3.fig.savefig('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures/ercot_max.png')

g4 = sns.FacetGrid(ercot_delta_min.reset_index(), col="Subregion", hue='Peak', col_wrap=4)
g4.map(plt.plot, 'Delta', 'value', linewidth=.5).add_legend()
plt.subplots_adjust(top=0.9)
g4.fig.suptitle('ERCOT Min')  # , y=1.05)
g4.fig.savefig('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures/ercot_min.png')
plt.show()
