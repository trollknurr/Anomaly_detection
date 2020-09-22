########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
from pathlib import Path
# additional
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
from scipy.io import wavfile

# original lib
from anomaly_detection import common as com
from anomaly_detection import keras_model
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load('anomaly_detection/config.yaml')
########################################################################


########################################################################
# load model
########################################################################

model_file = "anomaly_detection\{model}\model_{machine_type}.hdf5".format(
                                                        model=param["model_directory"],
                                                        machine_type=param["machine_type"]
                                                        )

model_dir = Path(os.path.dirname(__file__), model_file)


model = keras_model.load_model(model_dir)
#######################################################################


def get_anomaly(file_path):

    data = com.file_to_vector_array(file_path,
                                    n_mels=param["feature"]["n_mels"],
                                    frames=param["feature"]["frames"],
                                    n_fft=param["feature"]["n_fft"],
                                    hop_length=param["feature"]["hop_length"],
                                    power=param["feature"]["power"])

    errors = np.mean(np.square(data - model.predict(data)), axis=1)
    anomaly_score = np.mean(errors)
    
    if anomaly_score > param['threshold']:
        status = True
    else:
        status = False

    return status, anomaly_score