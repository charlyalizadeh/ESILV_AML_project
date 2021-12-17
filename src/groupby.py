import pandas as pd


data = pd.read_csv('../data/csv/train2_clean.csv')
data = data.groupby('fullVisitorId').agg(
        {
            'channelGrouping': 'max',
            'visitNumber': 'max',
            'visitStartTime': 'max',
            'source': 'last',
            'medium': 'last',
            'isTrueDirect': 'last',
            'referralPath': 'last',
            'hits': ['sum', 'mean', 'min', 'max', 'median'],
            'pageviews': ['sum', 'mean', 'min', 'max', 'median'],
            'bounces': ['sum', 'mean'],
            'newVisits': 'max',
            'transactionRevenue': 'sum',
            'continent': 'last',
            'subContinent': 'last',
            'country': 'last',
            'networkDomain': 'last',
            'browser': 'last',
            'operatingSystem': 'last',
            'isMobile': 'last',
            'deviceCategory': 'last',
            'page': 'last',
            'adNetworkType': 'last',
            'dayofweek': 'last',
            'month': 'last',
            'timeSpanSinceFirstVisit': 'last'
        }
)


for agg in ['sum', 'mean', 'min', 'max', 'median']:
    if agg in ('sum', 'mean'):
        data[f'bounces_{agg}'] = data['bounces'][agg]
    data[f'hits_{agg}'] = data['hits'][agg]
    data[f'pageviews_{agg}'] = data['pageviews'][agg]

data.drop(['hits', 'pageviews', 'bounces'], inplace=True, axis=1)
data.columns = data.columns.get_level_values(0)
data = data.reset_index()
data.drop('fullVisitorId', inplace=True, axis=1)
data.to_csv('../data/csv/processed.csv', index=False)
