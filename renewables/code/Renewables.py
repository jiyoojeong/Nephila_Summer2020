#!/usr/bin/env python
# coding: utf-8

# !pip install pandas
# !pip install matplotlib
# !pip install seaborn
# !pip install datetime

# In[1]:


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


# In[2]:



# ====== GLOBAL VARS ===== #
filedir = '/Users/jiyoojeong/desktop/C/Nephila/renewables/raw/forwards/spg/'




# In[26]:



# ====== FUNCTIONS ====== #


def combine_market(market):
    # list all in dir with name == to market

    lis = os.listdir(filedir)
    lis.remove('ERCOT')
    date = datetime.datetime.today()
    yesterday = (date - datetime.timedelta(days=1)).strftime("%m%d%Y")
    d = date.strftime("%m%d%Y")
    if True:#market + '_' + d + '.csv' not in os.listdir(filedir + 'ERCOT'):

        # print(d)
        print('Market: ' + market + '  Day: ' + d)
        big_one = pd.DataFrame()
        count = 0
        errors = 0
        for f in lis:
            try:
                if re.search(market, f) and not re.search('META', f):
                    print('----- new -----')
                    df = pd.read_csv(filedir + f, engine='python')
                    print('df read')
                    #print(f, df)
                    #print(df['Unnamed: 0.1'])
                    #print('helo')
                    #print('helllooo')
                    region = df.iloc[1, 1]
                    #print(region)
                    #break
                    region = region.split(':')[1].strip()
                    peak = df.iloc[3, 1]
                    peak = peak.split(':')[1].strip()
                    as_of_date = df.iloc[5, 1]  # accounts for weekend dates or days data is not available already
                    as_of_date = as_of_date.split(':')[1].strip()
                    headers = df.iloc[7, :].values
                    headers = np.delete(headers, 0)
                    print('got all first 7 rows')

                    # print(region, peak, as_of_date, headers, type(region), type(headers))
                    #print(df.columns)
                    df = df.drop(df.index[[0, 1, 2, 3, 4, 5, 6, 7]])
                    df = df.drop(columns=[df.columns[0]])
                    print('dropped columns')

                    #print(df.columns)
                    # change df to add new categorical parameters
                    df.columns = headers
                    #print(df.columns)
                    #print(df)
                    #break
                    df['AsOfDate'] = as_of_date

                    df['Peak'] = peak

                    df['Region'] = region

                    print('set new columns')

                    # print(df)
                    # big_one.columns = headers
                    big_one = big_one.append(df)
                    print('append to bigone')
                    # print(big_one)
            except:
                print('no settings like this one.', f)
                try:
                    print(df.iloc[6, :])
                except:
                    print(df)
                errors += 1
                #region2 = df.iloc[2, 1]
                #print(region2)
                #print(df.iloc[1, 1])
                ft = re.search(r'[0-9]{8}', f).span()
                wrongdate = f[ft[0]:ft[1]]
                print(wrongdate)
                print(datetime.datetime.strptime(wrongdate, '%m%d%Y').weekday())
                #break
            count += 1
            if count % 1000 == 0:
                # print(df)
                print(d)
                print(big_one.tail(10))


        os.chdir(filedir)
        # print(os.listdir(filedir))
        if market not in os.listdir(filedir):
            os.makedirs(market)
        bigfilename = filedir + market + '/' + market + '_' + str(d) + '.csv'
        big_one.to_csv(bigfilename)
        print('number errored:', errors)
        return bigfilename
    return filedir + market + '/' + market + '_' + d + '.csv'


def atc(on, off):
    # print(on[0], off[0])
    # print(type(on[0]), type(off[0]))
    b = (lambda x, y: (16.0 / 24) * x + (8.0 / 24 * 5 / 7 + 24 / 24 * 2 / 7) * y)(on.values, off.values)
    return b



# In[27]:



# ====== ERCOT ====== #
ercot_files = os.listdir(filedir + 'ERCOT')
# print(ercot_files)
#ercot_file = filedir + 'ERCOT/' + 'ERCOT_07282020.csv'  # combine_market('ERCOT')
ercot_file = combine_market('ERCOT')
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
sus = ercot.loc[ercot['Term'] == datetime.datetime.strptime('2019-08-01', '%Y-%m-%d'), : ]
sus = sus.loc[sus['AsOfDate'] == datetime.datetime.strptime('2019-08-19', '%Y-%m-%d'), :]


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
ercot_atc['Region'] = 'ERCOT'
ercot = ercot[ercot_atc.columns]
ercot = ercot.append(ercot_atc).sort_values(by=['AsOfDate', 'Term', 'Peak']).drop_duplicates().dropna()


# In[28]:


ercot[ercot['West Zone'] > 100]['Term'].dt.month.value_counts()
# months with forward prices greater than 100
# these are all summer months


# In[29]:


ercot.to_csv('ercot_all_ATC2.csv')


# In[19]:


ercot['Peak'].value_counts()


# In[16]:



''' Now we want to get the value difference from the historical to the predicted.'''
''' 
TODO:
- import the historical prices
- match each term date to each historical date
- find the difference
- use this difference as the new values for delta
'''

ERCOT_RT = pd.read_csv('/Users/jiyoojeong/desktop/C/Nephila/renewables/raw/reals/ERCOT/ERCOT_RT.csv')
ERCOT_RT['date'] = pd.to_datetime(ERCOT_RT['date'])
ERCOT_RT['price'] = ERCOT_RT['price'].astype(float)
print(ERCOT_RT)

ercot_zones = pd.melt(ercot,id_vars=['AsOfDate', 'Term', 'Peak'],
                           value_vars=zones,
                           var_name='Subregion')
#print(ercot_zones)
# ercot_rt_reshaped = ERCOT_RT.pivot(index='date', columns=['Peak', 'Subregion'])
# print(ercot_rt_reshaped)

# ercot = ercot.loc[['Houston Zone', 'North Zone', 'South Zone', 'West Zone']]
# ercot_early = ercot.append(ercot_atc).sort_values(by=['AsOfDate', 'Term', 'Peak'])
# TODO: Some magic and find the differences between prices for each term, peak, and region.
ercot_zones.sort_values('value', ascending=False).head(20)


# In[17]:



ercot_rt_h = ERCOT_RT.merge(ercot_zones, left_on=['date', 'Peak', 'Subregion'], right_on=['Term', 'Peak', 'Subregion'], how='inner')
#print(ercot_rt_h.head())

ercot_rt_h['Prediction Loss'] = ercot_rt_h['price'] - ercot_rt_h['value']
ercot_rt_h.sort_values('Prediction Loss')

ercot_rt_h.to_csv('realtime_forwards_ercot.csv')


# In[51]:



ercot_delta = ercot_rt_h.copy()
# ercot_delta_early = ercot_early.copy()
# monthly window averages
# ercot_delta = ercot.sort_values(by=['Term', 'AsOfDate']).rolling(30, min_periods=1).mean()

# day differences from predicted term date and the as of date
ercot_delta['Time Delta'] = ercot_delta['Term'] - ercot_delta['AsOfDate']
# ercot_delta_early['Time Delta'] = ercot_delta_early['Term'] - ercot_delta_early['AsOfDate']
# print(ercot_delta.head())
# drop index to make uniform
ercot_delta.reset_index(drop=True, inplace=True)
# melt columns so that the remaining columns are Delta Peak Subregion Value

#real_ercot = ercot_delta.loc[ercot_delta['Term'] == datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')]
#real_ercot = ercot_delta.copy() # LOOK AT 10 YEARS
real_ercot = ercot_delta[ercot_delta['Time Delta'] <= datetime.timedelta(days=365)] #LOOKING ONLY AT ONE YEAR

real_ercot


# In[52]:


real_ercot[real_ercot['Prediction Loss']>70]['Term'].value_counts()


# In[53]:


# may also need to look at https://stackoverflow.com/questions/55403008/pandas-partial-melt-or-group-melt

ercot_delta_melt = real_ercot.drop_duplicates()#.drop(columns=['date', 'Term', 'AsOfDate'])
#print(ercot_delta_melt.sort_values(['Time Delta', 'Subregion', 'Peak'], ascending=True).tail(20))
# group column 'Delta' and find statistics

# mean
ercot_delta_avg = ercot_delta_melt.groupby(['Time Delta', 'Peak', 'Subregion']).mean()
#print(ercot_delta_avg)
# standard deviation
ercot_delta_sd = ercot_delta_melt.groupby(['Time Delta', 'Peak', 'Subregion']).std()
# max
ercot_delta_max = ercot_delta_melt.groupby(['Time Delta', 'Peak', 'Subregion']).max()
# min
ercot_delta_min = ercot_delta_melt.groupby(['Time Delta','Peak', 'Subregion']).min()


# In[54]:


#ercot_delta_avg, ercot_delta_max, ercot_delta_min
ercot_delta_max


# In[55]:


#get_ipython().magic(u'matplotlib inline')


# ## PLOT 1 - MEAN
# 

# In[56]:


s_in_yr = 60*60*24*365
def timeTicks(s, pos):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    return str(s)
                                                                                                                                                                                                        
formatter = matplotlib.ticker.FuncFormatter(timeTicks)  


# In[57]:


# os.chdir('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/')
#if 'figures' not in os.listdir('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/'):
#    os.makedirs('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures')

# plot to see
ercot_delta_avg = ercot_delta_avg.reset_index()
print(type(ercot_delta_avg['Time Delta'][0]))
ercot_delta_avg['Time Delta'] = ercot_delta_avg['Time Delta'].dt.days
#print(ercot_delta_avg)
g = sns.FacetGrid(ercot_delta_avg, col="Subregion", hue='Peak', col_wrap=1, aspect=4)
g.map(plt.plot, 'Time Delta', 'Prediction Loss', linewidth=.5).add_legend()
plt.subplots_adjust(top=.9)
g.fig.suptitle('ERCOT Mean Prediction Loss for Forward Terms for January 2020')  # , y=1.05)
# g.fig.title('Ercot Means')
plt.gca().invert_xaxis()


for ax in g.axes.flatten():
    ax.xaxis.set_visible(True)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    
g.fig.savefig('/Users/jiyoojeong/desktop/C/Nephila/renewables/figures/ercot_means_Jan2020.png')



# In[58]:


# os.chdir('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/')
#if 'figures' not in os.listdir('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/'):
#    os.makedirs('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures')

# plot to see
g = sns.FacetGrid(ercot_delta_avg, col="Subregion", hue='Peak', col_wrap=4)
g.map(plt.hist,'Prediction Loss', alpha=.6, rwidth=1).add_legend()
plt.subplots_adjust(top=.8)
g.fig.suptitle('ERCOT Means Prediction Loss Distribution 1 Year')  # , y=1.05)
# g.fig.title('Ercot Means')
#plt.gca().invert_xaxis()
g.fig.savefig('/Users/jiyoojeong/desktop/C/Nephila/renewables/figures/ercot_mean_hist_1y.png')



# ## PLOT 2 - SD
# 

# In[59]:


ercot_delta_sd = ercot_delta_sd.reset_index()
ercot_delta_sd['Time Delta'] = ercot_delta_sd['Time Delta'].dt.days
#print(ercot_delta_avg)
g2 = sns.FacetGrid(ercot_delta_sd, col="Subregion", hue='Peak', col_wrap=1, aspect=4)
g2.map(plt.plot, 'Time Delta', 'Prediction Loss', linewidth=.5).add_legend()
plt.subplots_adjust(top=.9)

#g2.fig.suptitle('ERCOT Standard Deviation Prediction Loss Over One Year Delta Time Away')  # , y=1.05)
g2.fig.suptitle('ERCOT Standard Deviation Prediction Loss January 2020')  # , y=1.05)
plt.gca().invert_xaxis() # Invert for time delta plot

for ax in g2.axes.flatten():
    ax.xaxis.set_visible(True)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)

#g2.fig.savefig('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures/ercot_sd_deltapredict.png')
g2.fig.savefig('/Users/jiyoojeong/desktop/C/Nephila/renewables/figures/ercot_sd_Jan2020.png')
g2


# In[60]:


g2 = sns.FacetGrid(ercot_delta_sd, col="Subregion", hue='Peak', col_wrap=4)
#g2.map(plt.plot, 'Time Delta', 'Prediction Loss', linewidth=.5).add_legend()
g2.map(plt.hist,'Prediction Loss', alpha=.6, rwidth=1).add_legend()
plt.subplots_adjust(top=.8)
g2.fig.suptitle('ERCOT Standard Deviation Prediction Loss Distribution 1 Year')  # , y=1.05)
#plt.gca().invert_xaxis() # Invert for time delta plot

#g2.fig.savefig('/Users/jiyoojeong/desktop/C/raw/forwards/spg/ERCOT/figures/ercot_sd_deltapredict.png')
g2.fig.savefig('/Users/jiyoojeong/desktop/C/Nephila/renewables/figures/ercot_sd_hist_1y.png')
g2


# ## PLOT 3 MAX

# In[61]:


ercot_delta_max = ercot_delta_max.reset_index()
ercot_delta_max['Time Delta'] = ercot_delta_max['Time Delta'].dt.days
#print(ercot_delta_avg)
g3 = sns.FacetGrid(ercot_delta_max, col="Subregion", hue='Peak', col_wrap=2, aspect=2)
g3.map(plt.plot, 'Time Delta', 'Prediction Loss', linewidth=.5).add_legend()
plt.subplots_adjust(top=.9)

g3.fig.suptitle('ERCOT Max Prediction Loss for 1 Year')  # , y=1.05)
plt.gca().invert_xaxis()

g3.fig.savefig('/Users/jiyoojeong/desktop/C/Nephila/renewables/figures/ercot_max_1y.png')

g3


# ## PLOT 4 MIN

# In[62]:


ercot_delta_min = ercot_delta_min.reset_index()
ercot_delta_min['Time Delta'] = ercot_delta_min['Time Delta'].dt.days
#print(ercot_delta_avg)
g4 = sns.FacetGrid(ercot_delta_min, col="Subregion", hue='Peak', col_wrap=2, aspect=2)
g4.map(plt.plot, 'Time Delta', 'Prediction Loss', linewidth=.5).add_legend()
plt.subplots_adjust(top=.9)

g4.fig.suptitle('ERCOT Min Prediction Loss for 1 Year')  # , y=1.05)
plt.gca().invert_xaxis()
    
g4.fig.savefig('/Users/jiyoojeong/desktop/C/Nephila/renewables/figures/ercot_min_1y.png')

g4


# In[63]:


#plt.show()


# ## PLOT 5 - TEST DEALS

# In[64]:


# use ercot_rt_h
ercot_rt_h


# In[65]:


# ONLY LOOK AT 
# ATC, 
# JUNE as of date before June 18, 2014 (one year prior to first as of date), 
# and only June Term Dates

rel_cols = ['AsOfDate', 'Peak', 'Term', 'Subregion', 'value', 'price', 'Prediction Loss'] 
#value is forward, price is real time

ercot_rt_h_june = ercot_rt_h.loc[(ercot_rt_h['Term'] > '2015-12-31')&                                 (ercot_rt_h['AsOfDate'] <= '2015-12-31')                                 &(ercot_rt_h['AsOfDate'] > '2015-01-01')                                 &(ercot_rt_h['Peak'] == 'ATC'), rel_cols]
ercot_rt_h_june['Term Month'] = ercot_rt_h_june['Term'].dt.month
ercot_rt_h_june = ercot_rt_h_june[ercot_rt_h_june['Term Month'] == 1]
ercot_rt_h_june['Term Year'] = ercot_rt_h_june['Term'].dt.year

ercot_rt_h_means = ercot_rt_h_june.groupby(['AsOfDate', 'Subregion', 'Term', 'Term Year']).mean().reset_index()
ercot_rt_h_means = ercot_rt_h_means.rename(columns={'value': 'Forward Price', 'price': 'Real Time Price'})


# In[66]:


ercot_rt_h_means#[(ercot_rt_h_means['Term Year'] == 2014) & (ercot_rt_h_means['Subregion'] == 'Houston Zone')]


# In[67]:


# plot to see
g5 = sns.FacetGrid(ercot_rt_h_means, col="Subregion", hue='Term Year', col_wrap=1, aspect=7, palette='Blues_d')
g5.map(plt.plot, 'AsOfDate', 'Forward Price', linewidth=1).add_legend()
plt.subplots_adjust(top=.9)
#g.figure(figsize=(10, 4))
g5.fig.suptitle('ERCOT January Term Forward Values over one year\'s time in 2014 AsOfDates')  # , y=1.05)
# g.fig.title('Ercot Means')
#plt.gca().y_label('Prices')
#g5.fig.yaxis(label='Prices')
#plt.gca().invert_xaxis()
g.fig.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_jan_test_means_vals_scatter.png')


# In[68]:


# plot to see
ercot_rt_h_means['AsOfMonth'] = ercot_rt_h_means['AsOfDate'].dt.month
g6 = sns.FacetGrid(ercot_rt_h_means, col="Term Year", col_wrap=4, palette='RdBu')
g6.map(plt.hist, 'Forward Price', linewidth=.5, alpha=.5).add_legend()
plt.subplots_adjust(top=.8)
#g.figure(figsize=(10, 4))
g6.fig.suptitle('Distribution of ERCOT June Term Forward Values')  # , y=1.05)
# g.fig.title('Ercot Means')
#plt.gca().invert_xaxis()
g6.fig.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_june_test_means_hist_vals.png')


# In[69]:


# plot to see
consolidated_rt_h_means = ercot_rt_h_means.groupby(['AsOfDate', 'Term', 'Term Year', 'Term Month']).mean().reset_index()
consolidated_rt_h_m_melt = pd.melt(consolidated_rt_h_means,id_vars=['AsOfDate', 'Term', 'Term Year'],
                           value_vars=['Forward Price', 'Prediction Loss'],
                           var_name='metric')
#print(consolidated_rt_h_m_melt)
g7 = sns.FacetGrid(consolidated_rt_h_m_melt, col="metric", hue='Term Year', col_wrap=1, aspect=4, palette='Blues_d')
g7.map(plt.plot, 'AsOfDate', 'value', linewidth=1).add_legend()
plt.subplots_adjust(top=.8)
#g.figure(figsize=(10, 4))
g7.fig.suptitle('ERCOT Jan Term Forwards over one year\'s time in AsOfDates Mean Over All Subregions')  # , y=1.05)
# g.fig.title('Ercot Means')
#plt.gca().invert_xaxis()
g7.fig.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_test_mean_jan_all_markets.png')


# In[81]:


ercot_rt_h_means
consolidated_rt_h_means


# In[98]:


# plot to see
consolidated_rt_h_means = ercot_rt_h_means.groupby(['AsOfDate', 'Term', 'Term Year', 'Term Month']).mean().reset_index()
consolidated_rt_h_m_melt = pd.melt(consolidated_rt_h_means,id_vars=['AsOfDate', 'Term', 'Term Year'],
                           value_vars=['Forward Price', 'Prediction Loss'],
                           var_name='metric')
consolidated_means = consolidated_rt_h_m_melt.copy()
consolidated_rt_h_m_melt=consolidated_rt_h_m_melt[consolidated_rt_h_m_melt['metric'] == 'Forward Price']
consolidated_means_fp = consolidated_rt_h_m_melt.copy()
#print(consolidated_rt_h_m_melt)
g7 = sns.FacetGrid(consolidated_rt_h_m_melt, col="metric", hue='Term Year', height=10, aspect=2, palette='Blues_d')
g7.map(plt.plot, 'AsOfDate', 'value', linewidth=1).add_legend()
plt.subplots_adjust(top=.8)
#g.figure(figsize=(10, 4))
g7.fig.suptitle('ERCOT Jan Term Forwards over one year\'s time in AsOfDates All Subregions')  # , y=1.05)
# g.fig.title('Ercot Means')
#plt.gca().invert_xaxis()
only_rt = consolidated_rt_h_means[['AsOfDate', 'Real Time Price', 'Term Year']].drop_duplicates()
sns.lineplot(data=only_rt, x='AsOfDate', y='Real Time Price', hue='Term Year', palette='Greens_r')
#g7.fig.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_meansFV_test_jan2015.png')


# In[72]:


# plot to see
consolidated_rt_h_means = ercot_rt_h_means.groupby(['AsOfDate', 'Term', 'Term Year', 'Term Month']).mean().reset_index()
consolidated_rt_h_m_melt = pd.melt(consolidated_rt_h_means,id_vars=['AsOfDate', 'Term', 'Term Year'],
                           value_vars=['Forward Price', 'Prediction Loss'],
                           var_name='metric')
consolidated_means = consolidated_rt_h_m_melt.copy()
consolidated_rt_h_m_melt=consolidated_rt_h_m_melt[consolidated_rt_h_m_melt['metric'] == 'Prediction Loss']
consolidated_means_pl = consolidated_rt_h_m_melt.copy()
#print(consolidated_rt_h_m_melt)
g7 = sns.FacetGrid(consolidated_rt_h_m_melt, col="metric", hue='Term Year', height=10, aspect=2, palette='Reds_d')
g7.map(plt.plot, 'AsOfDate', 'value', linewidth=1).add_legend()
plt.subplots_adjust(top=.8)
#g.figure(figsize=(10, 4))
g7.fig.suptitle('ERCOT Jan Term Forwards over one year\'s time in AsOfDates All Subregions')  # , y=1.05)
# g.fig.title('Ercot Means')
#plt.gca().invert_xaxis()
g7.fig.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_meansPL_test_jan2015.png')


# In[73]:


# plot to see
consolidated_rt_h_sds = ercot_rt_h_means.groupby(['AsOfDate', 'Term', 'Term Year', 'Term Month']).std().reset_index()
consolidated_rt_h_m_melt = pd.melt(consolidated_rt_h_sds,id_vars=['AsOfDate', 'Term', 'Term Year'],
                           value_vars=['Forward Price', 'Prediction Loss'],
                           var_name='metric')
#print(consolidated_rt_h_m_melt)
consolidated_sds= consolidated_rt_h_m_melt.copy()
consolidated_rt_h_m_melt=consolidated_rt_h_m_melt[consolidated_rt_h_m_melt['metric'] == 'Forward Price']
consolidated_sds_FV = consolidated_rt_h_m_melt.copy()

g7 = sns.FacetGrid(consolidated_rt_h_m_melt, col="metric", hue='Term Year', col_wrap=1, aspect=3, palette='Greens_d')
g7.map(plt.plot, 'AsOfDate', 'value', linewidth=1, label='Term Year').add_legend()
plt.subplots_adjust(top=.8)
#g.figure(figsize=(10, 4))
g7.fig.suptitle('ERCOT Standard Deviation Jan Term Forwards over one year\'s time in AsOfDates over All Subregions')  # , y=1.05)
# g.fig.title('Ercot Means')
#plt.gca().invert_xaxis()
g7.fig.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_test_SD_Jan_all_markets.png')


# In[74]:


only_vals = consolidated_sds[consolidated_sds['metric']=='Prediction Loss'].rename(columns={'value': 'SD'})
#plt.plot('AsOfDate','SD',data=only_vals, label='Term', linewidth='.5')

sns.set(rc={'figure.figsize':(16,8)})
sns.set_style("whitegrid")
a = sns.lineplot(data=only_vals, x='AsOfDate', y='SD', hue='Term Year', palette='Greens')
plt.title('Standard Deviation of Forward Values for June 2020 Over One Year\'s Time')
plt.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_test_SDPL_JAN_line.png')


# # plot to see
# g8 = sns.FacetGrid(consolidated_rt_h_m_melt, col="metric", col_wrap=1, aspect=4, palette='RdBu')
# g8.map(plt.hist, 'value', alpha=.5).add_legend()
# plt.subplots_adjust(top=.8)
# #g.figure(figsize=(10, 4))
# g8.fig.suptitle('Distribution of ERCOT June Term Forward Prediction Loss')  # , y=1.05)
# # g.fig.title('Ercot Means')
# #plt.gca().invert_xaxis()
# 
# for ax in g8.axes.flatten():
#     ax.tick_params(labelbottom=True)
#     
# g8.fig.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_means_test_hist.png')
# 

# Can we take the P90/worst case/best case/expected forward price based off this volatility, and use Prius to price them? Hint: we can use the Prius API “Imposed Intermediate Results” function to impose a forward curve

# In[134]:


ercot_rt_h_means.groupby(['AsOfDate', 'Term', 'Term Year', 'Term Month']).count()


# In[140]:


# when do we usually price or buy a deal?
# that way I can try to predict the best/worst case from that as of date
range1 = ['2015-1-1','2015-3-31']
range2 = ['2015-6-1', '2015-9-30']
range3 = ['2015-10-31', '2015-12-31']

def inrangemax(df, r):
    return df.loc[(df['AsOfDate']>=r[0]) & (df['AsOfDate']<=r[1]), 'Prediction Loss'].min()

def inrangemin(df, r):
    return df.loc[(df['AsOfDate']>=r[0]) & (df['AsOfDate']<=r[1]), 'Prediction Loss'].max()


date1 = consolidated_rt_h_means.loc[(inrangemin(consolidated_rt_h_sds, range1) == consolidated_rt_h_sds['Prediction Loss']) &                           (consolidated_rt_h_sds['AsOfDate'] >= range1[0]) & (consolidated_rt_h_sds['AsOfDate'] <= range1[1]),
                           'AsOfDate'].values[0]
date2 = consolidated_rt_h_means.loc[(inrangemin(consolidated_rt_h_sds, range2) == consolidated_rt_h_sds['Prediction Loss']) &                           (consolidated_rt_h_sds['AsOfDate'] >= range2[0]) & (consolidated_rt_h_sds['AsOfDate'] <= range2[1]),
                           'AsOfDate'].values[0]
date3 = consolidated_rt_h_means.loc[(inrangemin(consolidated_rt_h_sds, range3) == consolidated_rt_h_sds['Prediction Loss']) &                           (consolidated_rt_h_sds['AsOfDate'] >= range3[0]) & (consolidated_rt_h_sds['AsOfDate'] <= range3[1]),
                           'AsOfDate'].values[0]

date1, date2, date3


# In[141]:


#date1 = datetime.datetime.strptime('2015-11-30', '%Y-%m-%d')# low vol at year end
#date2 = datetime.datetime.strptime('2015-01-30', '%Y-%m-%d') # high vol at year start
#date3 = datetime.datetime.strptime('2015-8-31', '%Y-%m-%d') # high vol here

date1df = consolidated_means[consolidated_means['AsOfDate'] == date1]

date2df = consolidated_means[consolidated_means['AsOfDate'] == date2]

date3df = consolidated_means[consolidated_means['AsOfDate'] == date3]
#consolidated_rt_h_means


# In[142]:


def highlight_max1(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    losses = date1pivot['Prediction Loss']
    #print(losses)
    is_max = np.abs(losses) == np.abs(losses).max()
    colors =  ['background-color: red; color: white' if is_max[i] else '' for i in np.arange(0, len(is_max))]
    is_min = np.abs(losses) == np.abs(losses).min()
    #print(is_max)
    colors = ['background-color: green; color: white' if is_min[i] else colors[i] for i in np.arange(0, len(is_min))]
    return colors

def highlight_max2(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    losses = date2pivot['Prediction Loss']
    #print(losses)
    is_max = np.abs(losses) == np.abs(losses).max()
    colors =  ['background-color: red; color: white' if is_max[i] else '' for i in np.arange(0, len(is_max))]
    is_min = np.abs(losses) == np.abs(losses).min()
    #print(is_max)
    colors = ['background-color: green; color: white' if is_min[i] else colors[i] for i in np.arange(0, len(is_min))]
    return colors

def highlight_max3(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    losses = date3pivot['Prediction Loss']
    #print(losses)
    is_max = np.abs(losses) == np.abs(losses).max()
    colors =  ['background-color: red; color: white' if is_max[i] else '' for i in np.arange(0, len(is_max))]
    is_min = np.abs(losses) == np.abs(losses).min()
    #print(is_max)
    colors = ['background-color: green; color: white' if is_min[i] else colors[i] for i in np.arange(0, len(is_min))]
    return colors


# In[143]:


date2df


# In[144]:


date1pivot = date1df.pivot(index=['AsOfDate', 'Term'], columns='metric', values='value')#[['Forward Price', 'Prediction Loss']]

date2pivot = date2df.pivot(index=['AsOfDate', 'Term'], columns='metric', values='value')#[['Forward Price', 'Prediction Loss']]

date3pivot = date3df.pivot(index=['AsOfDate', 'Term'], columns='metric', values='value')#[['Forward Price', 'Prediction Loss']]

date3pivot


# In[145]:


#cm = sns.light_palette("red", as_cmap=True)

s1 = date1pivot.style.apply(highlight_max1, subset=['Forward Price']).bar(subset=['Prediction Loss'], align='mid', color=['#ffe0e0'])
s2 = date2pivot.style.apply(highlight_max2, subset=['Forward Price']).bar(subset=['Prediction Loss'], align='mid', color=['#ffe0e0'])
s3 = date3pivot.style.apply(highlight_max3, subset=['Forward Price']).bar(subset=['Prediction Loss'], align='mid', color=['#ffe0e0'])


# In[146]:


s1


# In[147]:


s2


# In[148]:


s3


# In[181]:


merged = pd.merge(date1pivot.reset_index(), date2pivot.reset_index(), on='Term')
merged = pd.merge(merged, date3pivot.reset_index(), on='Term')
merged['Prediction Loss Difference'] = merged['Prediction Loss_x'] - merged['Prediction Loss_y']
merged['Forward Price Difference'] = merged['Forward Price_x'] - merged['Forward Price_y']


# In[185]:


m #= merged[['Term', 'AsOfDate_x', 'AsOfDate_y', 'Forward Price_x', 'Forward Price_y', 'Forward Price Difference', 
           #'Prediction Loss_x', 'Prediction Loss_y', 'Prediction Loss Difference']]
m = merged.copy()


# In[186]:


m['AsOfDate_x'][0], m['Forward Price_x'].std(), m['AsOfDate_y'][0], m['Forward Price_y'].std(), m['AsOfDate'][0], m['Forward Price'].std()
# FV SD for as of 11/30 and 1/30 and 8/31 -- (3.2512674950773413, 2.262517477475999, 1.4240268646826395)

# Timestamp('2015-02-05 00:00:00'), 2.249679212141074
# Timestamp('2015-08-26 00:00:00'), 1.009897509870272 ** LEAST VOLATILE
# Timestamp('2015-11-06 00:00:00'), 2.2157614248590547


# In[189]:


m['AsOfDate_x'][0], m['Prediction Loss_x'].std(), m['AsOfDate_y'][0], m['Prediction Loss_y'].std(), m['AsOfDate'][0], m['Prediction Loss'].std()
# MEAN PL for 11/30 = -10.415069047619042
# MEAN PL 1/30 = -16.844378571428564
# MEAN PL 8/21 = -13.495640476190468

# MEAN PL
# Timestamp('2015-02-05 00:00:00'), -16.385878571428563,
# Timestamp('2015-08-26 00:00:00'), -13.777378571428565,
# Timestamp('2015-11-06 00:00:00'), -10.712021428571422 ** MOST ACCURATE

# SD PL
#Timestamp('2015-02-05 00:00:00'), 3.789348970044771, ** LEAST VOLATILE
#Timestamp('2015-08-26 00:00:00'), 4.385065675002104,
#Timestamp('2015-11-06 00:00:00'), 4.167841736891567)





# In[190]:


# plot to see
consolidated_rt_h_means = ercot_rt_h_means.groupby(['AsOfDate', 'Term', 'Term Year', 'Term Month']).mean().reset_index()
consolidated_rt_h_m_melt = pd.melt(consolidated_rt_h_means,id_vars=['AsOfDate', 'Term', 'Term Year'],
                           value_vars=['Forward Price', 'Prediction Loss'],
                           var_name='metric')
consolidated_means = consolidated_rt_h_m_melt.copy()
consolidated_rt_h_m_melt=consolidated_rt_h_m_melt[consolidated_rt_h_m_melt['metric'] == 'Forward Price']
consolidated_means_fp = consolidated_rt_h_m_melt.copy()
#print(consolidated_rt_h_m_melt)
g7 = sns.FacetGrid(consolidated_rt_h_m_melt, col="metric", hue='Term Year', height=10, aspect=2, palette='Blues_d')
g7.map(plt.plot, 'AsOfDate', 'value', linewidth=1).add_legend()
plt.subplots_adjust(top=.8)
#g.figure(figsize=(10, 4))
g7.fig.suptitle('ERCOT Jan Term Forwards over one year\'s time in AsOfDates All Subregions')  # , y=1.05)
# g.fig.title('Ercot Means')
#plt.gca().invert_xaxis()
only_rt = consolidated_rt_h_means[['AsOfDate', 'Real Time Price', 'Term Year']].drop_duplicates()
datelist = [date1, date2, date3]
fvs = consolidated_rt_h_means[consolidated_rt_h_means['AsOfDate'].isin(datelist)]
sns.scatterplot(data=fvs, x='AsOfDate', y='Forward Price', hue='AsOfDate', palette=['red', 'green', 'orange'])
#a.text(x='AsOfDate', y='Forward Price', s='Forward Price')
g7.fig.savefig('/Users/jiyoojeong/desktop/C/nephila/renewables/figures/ercot_meansFV_w_deals.png')


# In[ ]:




