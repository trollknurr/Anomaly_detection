import glob
import os
import pathlib
import sys
import tensorflow as tf
import librosa
import numpy as np
import argparse
import tempfile
import boto3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from scipy.io import wavfile
import keras
import common as com
import keras_model

param = com.yaml_load('config.yaml')

input_schema = Schema([ColSpec("string", "path")])
output_schema = Schema([ColSpec("float")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)



def list_to_vector_array(file_list, param):
    n_mels = param["feature"]["n_mels"]
    frames = param["feature"]["frames"]
    dims = n_mels * frames

    for idx in range(len(file_list)):
        vector_array = preprocess(file_list[idx], param)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def file_list_generator(target_dir, ext="wav"):
    training_list_path = os.path.abspath("{dir}/*.{ext}".format(dir=target_dir, ext=ext))
    files = sorted(glob.glob(training_list_path))
    return files


def preprocess(file_path, param):
    n_mels = param["feature"]["n_mels"]
    frames = param["feature"]["frames"]
    n_fft = param["feature"]["n_fft"]
    hop_length = param["feature"]["hop_length"]
    power = param["feature"]["power"]

    file_name = file_path
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    sr, y = wavfile.read(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T
    return vector_array

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

make_keras_picklable()

def make_train_dataset(files_dir, param):
    files = file_list_generator(files_dir)
    dataset = list_to_vector_array(files, param)
    return dataset


class AEA(mlflow.pyfunc.PythonModel):

    def __init__(self, files_dir, param, preprocess, make_train_dataset,s3):
        self.preprocess = preprocess
        setattr(tf.compat.v1.nn.rnn_cell.GRUCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.compat.v1.nn.rnn_cell.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.compat.v1.nn.rnn_cell.MultiRNNCell, '__deepcopy__', lambda self, _: self)
        self.s3 = s3
        dataset = make_train_dataset(files_dir, param)
        inputDim = param["feature"]["n_mels"] * param["feature"]["frames"]
        inputLayer = Input(shape=(inputDim,))

        h = Dense(128)(inputLayer)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(8)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(128)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(inputDim)(h)
        self.model = Model(inputs=inputLayer, outputs=h)
        self.param = param
        self.model.compile(**param["fit"]["compile"])
        self.model.summary()
        self.model.fit(dataset, dataset,
                       epochs=param["fit"]["epochs"],
                       batch_size=param["fit"]["batch_size"],
                       shuffle=param["fit"]["shuffle"],
                       validation_split=param["fit"]["validation_split"],
                       verbose=param["fit"]["verbose"])

    def predict(self, context, model_input):
        file_path = str(model_input["path"][0])
        # file_path_loc = './tmp/' + file_path
        # local_path = str(pathlib.Path(file_path_loc).parent.mkdir(parents=True, exist_ok=True))

        obj=self.s3.Bucket('testdata').Object(file_path).get()
        data = self.preprocess(obj['Body'], self.param)
        result = self.model.predict(data)
        errors = np.mean(np.square(data - result), axis=1)
        return np.mean(data)


if __name__ == '__main__':
    make_keras_picklable()
    aws_access_key_id='minio'
    aws_secret_access_key = 'miniostorage'
    endpoint_url = 'http://10.0.2.15:9000'
    mlflow.set_tracking_uri("http://localhost:5003")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
    # exp_id = mlflow.set_experiment("/ae_keras")
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()
    file_name = args.path
    s3=boto3.resource('s3',
                            endpoint_url=endpoint_url,
                            aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key)
    with mlflow.start_run(run_name="Mlflow_test") as run:
        modelV = AEA(file_name, param, preprocess, make_train_dataset,s3)
        # mlflow.pyfunc.log_model("model", python_model=model,
        #                         conda_env='mlflowtestconf.yaml',
        #                         signature=signature)
        mlflow.pyfunc.log_model(artifact_path="model", python_model=modelV, signature=signature, code_path=[__file__],conda_env='mlflowtestconf.yaml')
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.end_run()