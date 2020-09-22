import requests
import json
files = {'file': open('test_data/anomaly_defectid_1_id_freq_44.wav', 'rb')}

r = requests.post('http://127.0.0.1:5000/', files=files)

answ = r.text

print(json.loads(answ)['filename'])
print(json.loads(answ)['score'])
print(json.loads(answ)['status'])