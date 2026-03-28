import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/home/arshiya/realistic_log_project/dataset/structured_logs.csv', low_memory=False)
le = LabelEncoder()
for col in ['log_level','component','event_type','status']:
    df[col] = le.fit_transform(df[col].astype(str))

session_df = df.groupby('session_id').agg({'anomaly_label':'max'})
print('Total sessions    :', len(session_df))
print('Normal sessions   :', (session_df.anomaly_label==0).sum())
print('Anomaly sessions  :', (session_df.anomaly_label==1).sum())
print('Anomaly ratio     :', round((session_df.anomaly_label==1).mean()*100, 2), '%')

# Also check avg logs per session
logs_per_session = df.groupby('session_id').size()
print('Avg logs/session  :', round(logs_per_session.mean(), 1))
print('Max logs/session  :', logs_per_session.max())
print('Min logs/session  :', logs_per_session.min())
