import json

import requests

import common as com
param = com.yaml_load('/home/rnd/Anomaly_detection/config.yaml')

def preprocess(file_path):
    data = com.file_to_vector_array(file_path,
                                    n_mels=param["feature"]["n_mels"],
                                    frames=param["feature"]["frames"],
                                    n_fft=param["feature"]["n_fft"],
                                    hop_length=param["feature"]["hop_length"],
                                    power=param["feature"]["power"])
    return data
url = "http://127.0.0.1:5004/invocations"

data = {
    "path":  'anomaly_defectid_1_id_freq_44.wav'}
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
r = requests.post(url, data=json.dumps(data), headers=headers)
print(r.text)