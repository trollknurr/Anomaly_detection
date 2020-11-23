import mlflow
import os
import pandas as pd
from flask import Flask
from flask import request
app = Flask(__name__)
aws_access_key_id =  os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
endpoint_url =os.environ.get('MLFLOW_S3_ENDPOINT_URL')
path_to_model=os.environ.get('ARTIFACT_STORE')
mmodel=mlflow.pyfunc.load_model(path_to_model)
port=os.environ.get('SERVER_PORT')
host=os.environ.get('SERVER_HOST')

@app.route('/invocations', methods=['POST'])
def predict():
    df=pd.read_json(request.data,orient='split')
    res=mmodel.predict(df)
    return res

if __name__ == '__main__':
    app.run(debug=True,port=port,host=host)
