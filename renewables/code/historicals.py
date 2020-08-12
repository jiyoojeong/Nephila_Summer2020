import pandas as pd
import numpy as np
import datetime as datetime
'''
todo:
- download historicals
- calculate on peak and off peak and atc for monthly (average months)
- combine historicals
- save to some file

Recall that on-peak include hours 6-21 M-F and off-peak are hours 0-5 and 22-23 M-F as well as all weekend hours and holidays. Let me know if you have any questions.


Holidays in ERCOT (with 2020 dates as examples) are:
New Year's Day 1/1/2020
MLK Day 1/20/2020
Memorial Day 5/25/2020
Independence Day 7/3/2020
Labor Day 9/7/2020
Thanksgiving (Thursday) 11/26/2020
Thanksgiving (Friday) 11/27/2020
Christmas Eve 12/24/2020
Christmas 12/25/2020
'''

houston_real = pd.read_csv('/Users/jiyoojeong/desktop/C/raw/reals/ERCOT/ERCOT-Houston-RT_Price.csv')
north_real = pd.read_csv('/Users/jiyoojeong/desktop/C/raw/reals/ERCOT/ERCOT-North-RT_Price.csv')
south_real = pd.read_csv('/Users/jiyoojeong/desktop/C/raw/reals/ERCOT/ERCOT-South-RT_Price.csv')
west_real = pd.read_csv('/Users/jiyoojeong/desktop/C/raw/reals/ERCOT/ERCOT-West-RT_Price.csv')

holidays = pd.to_datetime(pd.Series(['1/1/2020', '1/20/2020', '5/25/2020', '7/4/2020', '9/7/2020', '9/7/2020',
                                     '12/24/2020', '12/25/2020']))

# how to account for thanksgiving
# third thurs and fri


def real(real):
    real['dateTimeUTC'] = pd.Series([pd.Timestamp(t) for t in real['dateTimeUTC']])
    real.loc[real['price'] == '-', 'price'] = np.nan
    real['price'] = real['price'].astype(float)
    real['date'] = real['dateTimeUTC'].dt.date
    not_holiday = [np.all((holidays - d)/datetime.timedelta(days=1) % 365 >= 1) for d in real['dateTimeUTC']]
    holiday = [np.any((holidays - d)/datetime.timedelta(days=1) % 365 < 1) for d in real['dateTimeUTC']]

    # ATC
    print('atc')
    means_atc = real.groupby('date').mean()
    means_monthly_atc = means_atc.rolling(30, min_periods=1).mean()
    means_monthly_atc['Peak'] = 'ATC'

    # days and hours ON PEAK
    print('on peak')
    means_on = real.loc[not_holiday, :]
    means_on = means_on.loc[real['dateTimeUTC'].dt.weekday < 5, :]
    means_on = means_on.loc[means_on['dateTimeUTC'].dt.hour.isin(np.arange(6, 22)), :]

    means_on = means_on.groupby('date').mean()

    means_monthly_on = means_on.rolling(4, min_periods=1).mean()
    means_monthly_on['Peak'] = 'On Peak'

    # days and hours OFF PEAK
    print('off peak')
    means_off_holidays = real.loc[holiday, :]

    means_off_weekends = real.loc[real['dateTimeUTC'].dt.weekday >= 5, :]

    means_off_regular = real.loc[not_holiday, :]
    means_off_regular = means_off_regular.loc[real['dateTimeUTC'].dt.weekday < 5, :]
    means_off_regular = means_off_regular.loc[~means_off_regular['dateTimeUTC'].dt.hour.isin(np.arange(6, 22)), :]
    means_off = means_off_regular.append(means_off_weekends).append(means_off_holidays)

    means_off = means_off.groupby('date').mean()

    means_monthly_off = means_off.rolling(4, min_periods=1).mean()
    means_monthly_off['Peak'] = 'Off Peak'

    return means_monthly_atc.append(means_monthly_on).append(means_monthly_off)


all_houston_RT = real(houston_real)
all_houston_RT['Subregion'] = 'Houston Zone'
#print(all_houston_RT)

all_north_RT = real(north_real)
all_north_RT['Subregion'] = 'North Zone'

all_south_RT = real(south_real)
all_south_RT['Subregion'] = 'South Zone'

all_west_RT = real(west_real)
all_west_RT['Subregion'] = 'West Zone'

ERCOT_RT = all_houston_RT.append(all_north_RT).append(all_south_RT).append(all_west_RT)

print(ERCOT_RT)

ERCOT_RT.to_csv('/Users/jiyoojeong/desktop/C/raw/reals/ERCOT/ERCOT_RT.csv')