import requests
import json
import glob
files=glob.glob('test_data/*.wav')
import time
while True:
    for file in list(files):
        print(file)
        filess = {'file': open(file, 'rb')}
#
        r = requests.post('http://0.0.0.0:8086/', files=filess)

        answ = r.text
        time.sleep(5)
