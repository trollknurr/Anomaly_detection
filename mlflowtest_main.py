import csv
import glob
import os

import numpy as np
import argparse
import boto3
from botocore.client import Config

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import common as com
import keras_model

param = com.yaml_load('config.yaml')

input_schema = Schema([ColSpec("string", "path"), ])
output_schema = Schema([ColSpec("float")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
s3 = boto3.resource('s3',
                    endpoint_url='http://localhost:9000',
                    aws_access_key_id='minio',
                    aws_secret_access_key='miniostorage')

def list_to_vector_array(file_list,
                         n_mels=param["feature"]["n_mels"],
                         frames=param["feature"]["frames"]):
    dims = n_mels * frames

    for idx in range(len(file_list)):
        vector_array = preprocess(file_list[idx])
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def file_list_generator(target_dir, ext="wav"):
    training_list_path = os.path.abspath("{dir}/*.{ext}".format(dir=target_dir, ext=ext))
    files = sorted(glob.glob(training_list_path))
    return files


def preprocess(file_path):
    data = com.file_to_vector_array(file_path,
                                    n_mels=param["feature"]["n_mels"],
                                    frames=param["feature"]["frames"],
                                    n_fft=param["feature"]["n_fft"],
                                    hop_length=param["feature"]["hop_length"],
                                    power=param["feature"]["power"])
    return data


def make_train_dataset(files_dir):
    files = file_list_generator(files_dir)
    dataset = list_to_vector_array(files)
    return dataset


class AEA(mlflow.pyfunc.PythonModel):

    def __init__(self, files_dir):
        dataset= make_train_dataset(files_dir)
        self.model = keras_model.get_model(param["feature"]["n_mels"] * param["feature"]["frames"])
        self.model.summary()
        self.model.compile(**param["fit"]["compile"])
        self.model.fit(dataset,dataset,
                            epochs=param["fit"]["epochs"],
                            batch_size=param["fit"]["batch_size"],
                            shuffle=param["fit"]["shuffle"],
                            validation_split=param["fit"]["validation_split"],
                            verbose=param["fit"]["verbose"])

    def predict(self, context, model_input):
        file_path = str(model_input["path"][0])
        local_path='/tmp/' + file_path
        s3.Bucket('testdata').download_file(file_path, local_path)
        data = preprocess(local_path)
        result = self.model.predict(data)
        return result

if __name__=='__main__':
    mlflow.set_tracking_uri("http://localhost:5003")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'miniostorage'
    #exp_id = mlflow.set_experiment("/ae_keras")
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    file_name = args.path

    with mlflow.start_run():
        mlflow.keras.autolog()
        model = AEA('/home/rnd/Anomaly_detection/train_data')
        # mlflow.pyfunc.log_model("model", python_model=model,
        #                         conda_env='mlflowtestconf.yaml',
        #                         signature=signature)
        mlflow.keras.log_model(model.model,'model',signature=signature)
