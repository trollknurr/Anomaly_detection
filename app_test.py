import requests
import json
files = {'file': open('anomaly_id_1_pejo_206_275.wav', 'rb')}


r = requests.post('http://192.168.1.144:8032/', files=files)

answ = r.text

print(json.loads(answ)['datetime'])
print(json.loads(answ)['score'])
print(json.loads(answ)['status'])
