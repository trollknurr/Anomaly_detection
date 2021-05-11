import requests
import json
files = {'file': open('anomaly_id_1_pejo_206_275.wav', 'rb')}


r = requests.post('detector:5000/', files=files)

answ = r.text

print(json.loads(answ)['datetime'])
print(json.loads(answ)['score'])
print(json.loads(answ)['status'])
