import pathlib

import common as com
import keras_model
import glob
import os

import numpy as np
import boto3
param = com.yaml_load('config.yaml')

s3 = boto3.resource('s3',
                    endpoint_url='http://10.0.2.15:9000',
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
dataset= make_train_dataset('/home/rnd/Anomaly_detection/test_data')
model = keras_model.get_model(param["feature"]["n_mels"] * param["feature"]["frames"])
model.summary()
model.compile(**param["fit"]["compile"])
model.fit(dataset,dataset,
                    epochs=param["fit"]["epochs"],
                    batch_size=param["fit"]["batch_size"],
                    shuffle=param["fit"]["shuffle"],
                    validation_split=param["fit"]["validation_split"],
                    verbose=param["fit"]["verbose"])

file_path = 'anomaly_defectid_1_id_freq_44.wav'
file_path_loc='./tmp/' + file_path
print(file_path_loc)
local_path=str(pathlib.Path(file_path_loc).parent.mkdir(parents=True, exist_ok=True))
s3.Bucket('testdata').download_file(file_path, local_path)
data = preprocess(local_path)
print(data.shape)
result = model.predict(data)
print(result)