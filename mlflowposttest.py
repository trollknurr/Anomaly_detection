import json

import requests

import common as com
import pandas as pd
param = com.yaml_load('/home/rnd/Anomaly_detection/config.yaml')

url = "http://127.0.0.1:5004/invocations"

# data = '[{{"path": {path},"n_mels": {n_mels},"frames": {frames},"n_fft": {n_fft},"hop_length": {hop_length},"power": {power}}}]'.format(
#     path="anomaly_defectid_1_id_freq_44.wav",
#     n_mels=param["feature"]["n_mels"],
#     frames=param["feature"]["frames"],
#     n_fft=param["feature"]["n_fft"],
#     hop_length=param["feature"]["hop_length"],
#     power=param["feature"]["power"])
path ='anomaly_defectid_1_id_freq_44.wav'
n_mels=param["feature"]["n_mels"]
frames=param["feature"]["frames"]
n_fft=param["feature"]["n_fft"]
hop_length=param["feature"]["hop_length"]
power=param["feature"]["power"]
columns=['path','n_mels','frames','n_fft','hop_length','power']
data=[[path,n_mels,frames,n_fft,hop_length,power]]
df=pd.DataFrame(data=data,columns=columns)
print(df)
# data = [{'path': path,'n_mels': n_mels,'frames': frames,'n_fft': n_fft,'hop_length': hop_length,'power': power}]
data=df.to_json(index=False,orient='split')
print(data)
headers = {'Content-type': 'application/json', 'format': 'pandas-records'}
r = requests.post(url, data=data, headers=headers)
print(r.text)
