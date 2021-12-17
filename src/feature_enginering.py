import pandas as pd

data = pd.read_csv('../data/csv/train2_clean.csv', dtype={'fullVisitorId': str})

if 'timeSpanSinceFirstVisit' not in data.columns:
    grouped = data.groupby('fullVisitorId').min()
    data['timeSpanSinceFirstVisit'] = data.apply(
            lambda x: x['visitStartTime'] - grouped.loc[x['fullVisitorId']]['visitStartTime'],
            axis=1
    )

data.to_csv('../data/csv/train2_clean.csv', index=False)
