import json
import requests
import pandas as pd

url = "http://127.0.0.1:5010/invocations"
col=['path']
data = [['anomaly_defectid_1_id_freq_44.wav']]
df=pd.DataFrame(data=data,columns=col)
data=df.to_json(index=False,orient='split')
print(data)
headers = {'Content-type': 'application/json', 'format': 'pandas-records'}
r = requests.post(url, data=data, headers=headers)
