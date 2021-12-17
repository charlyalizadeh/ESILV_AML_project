import pandas as pd

data = pd.read_csv('../data/csv/train2.csv')

# NaN in the column transactionRevenue is considered as no transaction, so equal to 0
data['transactionRevenue'].fillna(0, inplace=True)

# We also make this assumption for the following columns but we don't have any information
# to make a good decision
data['isTrueDirect'].fillna(0, inplace=True)
data['page'].fillna(0, inplace=True)
data['adNetworkType'].fillna(0, inplace=True)
data['newVisits'].fillna(0, inplace=True)
data['bounces'].fillna(0, inplace=True)
data['pageviews'].fillna(0, inplace=True)

# Drop unique value columns
unique_value_columns = [c for c in data.columns if len(data[c].unique()) <= 1]
data.drop(unique_value_columns, axis=1, inplace=True)

# Drop useless or not complete columns
data.drop(['metro', 'region', 'city', 'sessionId', 'visitId', 'adContent',
           'isVideoAd', 'slot', 'gclId', 'keyword', 'campaign'], axis=1, inplace=True)


# Convert categorical variables
for c in ['channelGrouping', 'source', 'medium', 'continent', 'subContinent',
          'country', 'browser', 'operatingSystem', 'deviceCategory', 'networkDomain',
          'referralPath', 'isTrueDirect', 'adNetworkType']:
    data[c] = data[c].astype('category')
    data[c] = data[c].cat.codes

# Convert boolean variables
for c in data.select_dtypes(['bool']).columns:
    data[c] = data[c].astype(int)

# Convert date datatypes 
data['date'] = data['date'].astype(str)
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

# Extract the day of week and the month (year is irrelevant because the data is only on one year)
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data.drop('date', axis=1, inplace=True)
data['fullVisitorId'] = data['fullVisitorId'].astype(str)
data.to_csv('../data/csv/train2_clean.csv', index=False)
