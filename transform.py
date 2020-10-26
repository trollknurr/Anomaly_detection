########################################################################
# import python-library
########################################################################
# default
import sys
import os
import glob
import json
import logging
import shutil
from random import shuffle
import datetime
# additional
import yaml
import librosa
import pandas as pd
from tqdm import tqdm
from scipy.io.wavfile import write
# original lib
import common
########################################################################

########################################################################
# configuration and logs
########################################################################
logging.basicConfig(
                    level=logging.DEBUG, 
                    filename='logs.log', 
                    format='%(asctime)s %(levelname)s:%(message)s'
                    )

param = common.yaml_load()

iter_normal = ['normal', 'anomaly']
iter_type = {'audio': 'wav', 'vibro': 'txt', 'modbus': 'csv'}
########################################################################

def file_name_list(dir, dir_name, prefix_normal, prefix_type, ext):
    """
    Return file name with extention in path
    """

    file_name_list = sorted(glob.glob("{dir}/{dir_name}/{prefix_normal}/{prefix_type}/*.{ext}".format(
                                                                                                        dir=dir,
                                                                                                        dir_name=dir_name,
                                                                                                        prefix_normal=prefix_normal,
                                                                                                        prefix_type=prefix_type,
                                                                                                        ext=ext)
                                                                                                        ))
    file_name_list = [os.path.split(file_name)[-1].split('.')[0] for file_name in file_name_list]

    return file_name_list

def generate_dataset(path):
    """
    Transform csv, modbus, wav info one json 
    """

    file_path_dict = {
                'normal': 
                        {
                        'audio': [],
                        'modbus': [],
                        'vibro': []
                        },
                'anomaly':
                        {
                        'audio': [],
                        'modbus': [],
                        'vibro': []
                        }
                }

    for status in iter_normal:
        print(f'Start generate {status} dataset')
        for key, value in iter_type.items():
            file_names = file_name_list(path, 'dirty', status, key, value)
            file_path_dict[status][key] = file_names

        filename_status = list(
                                set(file_path_dict[status]['audio']) & 
                                set(file_path_dict[status]['vibro']) & 
                                set(file_path_dict[status]['modbus'])
                            )
        shuffle(filename_status)
        for id, filename in enumerate(tqdm(filename_status), start=1):
            logging.debug(f"Start filtered {filename}")
            data = {
                'audio': {
                            'ys': [],
                            'framerate': None
                        },

                'modbus': {
                            'freq': []
                            },
                'datetime': None
                }

            data['datetime'] = filename

            ys, framerate = librosa.load("{dir}/{dir_name}/{prefix_normal}/{prefix_type}/{file}.{ext}".format(
                                                                                                        dir=param['data'],
                                                                                                        dir_name='dirty',
                                                                                                        prefix_normal=status,
                                                                                                        prefix_type='audio',
                                                                                                        file = filename,
                                                                                                        ext='wav')
                                                                                                        )
            if (len(ys) / (framerate)) < 10:
                logging.debug(f"{filename}: duration is not 10 sec! \n")
                continue

            data['audio']['ys'], data['audio']['framerate'] = ys[:framerate * 10].tolist(), framerate

            modbus = pd.read_csv("{dir}/{dir_name}/{prefix_normal}/{prefix_type}/{file}.{ext}".format(
                                                                                                    dir=param['data'],
                                                                                                    dir_name='dirty',
                                                                                                    prefix_normal=status,
                                                                                                    prefix_type='modbus',
                                                                                                    file = filename,
                                                                                                    ext='csv')
                                                                                                )
            data['modbus']['freq'] = modbus['specified_frequency'].values.tolist()

            mean_freq = round((modbus['specified_frequency'].values.mean()))

            filename_gen = f"{status}_id_{id}_freq_{mean_freq}"

            with open("{dir}/{dir_name}/{prefix_normal}/{filename}.{ext}".format(
                                                                                    dir=param['data'],
                                                                                    dir_name='clear',
                                                                                    prefix_normal=status,
                                                                                    filename=filename_gen,
                                                                                    ext='json'), 'w'
                                                                                ) as f:
                json.dump(data, f)

    print('Transform into JSON is ready')

def generate_dataset_wav(path):
    """
    Transform csv, modbus, wav info one json 
    """

    file_path_dict = {
                'normal': 
                        {
                        'audio': [],
                        'modbus': [],
                        'vibro': []
                        },
                'anomaly':
                        {
                        'audio': [],
                        'modbus': [],
                        'vibro': []
                        }
                }

    for status in iter_normal:
        print(f'Start generate {status} dataset')
        for key, value in iter_type.items():
            file_names = file_name_list(path, 'dirty', status, key, value)
            file_path_dict[status][key] = file_names

        filename_status = list(
                                set(file_path_dict[status]['audio']) & 
                                set(file_path_dict[status]['vibro']) & 
                                set(file_path_dict[status]['modbus'])
                            )
        shuffle(filename_status)
        for id, filename in enumerate(tqdm(filename_status), start=1):
            logging.debug(f"Start filtered {filename}")

            ys, framerate = librosa.load("{dir}/{dir_name}/{prefix_normal}/{prefix_type}/{file}.{ext}".format(
                                                                                                        dir=param['data'],
                                                                                                        dir_name='dirty',
                                                                                                        prefix_normal=status,
                                                                                                        prefix_type='audio',
                                                                                                        file = filename,
                                                                                                        ext='wav')
                                                                                                        )
            if (len(ys) / (framerate)) < 10:
                logging.debug(f"{filename}: duration is not 10 sec! \n")
                continue

            modbus = pd.read_csv("{dir}/{dir_name}/{prefix_normal}/{prefix_type}/{file}.{ext}".format(
                                                                                                    dir=param['data'],
                                                                                                    dir_name='dirty',
                                                                                                    prefix_normal=status,
                                                                                                    prefix_type='modbus',
                                                                                                    file = filename,
                                                                                                    ext='csv')
                                                                                                )
            mean_freq = int((modbus['specified_frequency'].values.mean()))

            defect_id = round(modbus['defect_id'].iloc[0], 2)
            filename_gen = f"{status}_defectid_{defect_id}_id_{id}_freq_{mean_freq}"

            y, sr = ys[:framerate * 10], framerate
            #print( {'id': int(id), 'framerate': int(sr), 'ys': list(y), 'datetime': datetime.datetime.now(), 'defect_id': int(defect_id), 'freq': mean_freq})
            idobj = mongo_test.insert_document(DB.dbconfig.COLLECTION_NAME, {'id': int(id), 'framerate': int(sr), 'ys': list(y.astype(float)), 'datetime': datetime.datetime.now(), 'defect_id': int(defect_id), 'freq': mean_freq})

            id_return_list.append(idobj)
            
            write("{dir}/{dir_name}/{prefix_normal}/{filename}.{ext}".format(
                                                                            dir=param['data'],
                                                                            dir_name='clear',
                                                                            prefix_normal=status,
                                                                            filename=filename_gen,
                                                                            ext='wav'
                                                                            ), sr, y)

    print('Transform into WAV is ready')

def train_test_split_wav():

    try:
        shutil.rmtree('dev_data/engine/train/')
    except:
        pass
    try:
        shutil.rmtree('dev_data/engine/test/')
    except:
        pass

    os.mkdir('dev_data/engine/train/') 
    os.mkdir('dev_data/engine/test/') 

    file_path_list_anomaly = sorted(glob.glob("{dir}/{dir_name}/{prefix_normal}/*.{ext}".format(
                                                                                                dir='data',
                                                                                                dir_name='clear',
                                                                                                prefix_normal='anomaly',
                                                                                                ext='wav')
                                                                                                ))

    file_name_list_anomaly = [os.path.split(file_name)[-1].split('.')[0] for file_name in file_path_list_anomaly]

    for path, name in tqdm(zip(file_path_list_anomaly, file_name_list_anomaly)):
        shutil.move(path, 'dev_data/engine/test/' + name + '.wav')
    
    file_path_list_normal = sorted(glob.glob("{dir}/{dir_name}/{prefix_normal}/*.{ext}".format(
                                                                                                dir='data',
                                                                                                dir_name='clear',
                                                                                                prefix_normal='normal',
                                                                                                ext='wav')
                                                                                                ))

    file_name_list_normal = [os.path.split(file_name)[-1].split('.')[0] for file_name in file_path_list_normal]

    thld = len(file_name_list_normal) // 3
    for path, name in tqdm(zip(file_path_list_normal[:thld], file_name_list_normal[:thld])):
        shutil.move(path, 'dev_data/engine/test/' + name + '.wav')

    for path, name in tqdm(zip(file_path_list_normal[thld:], file_name_list_normal[thld:])):
        shutil.move(path, 'dev_data/engine/train/' + name + '.wav')

if __name__ == "__main__":

    path = common.yaml_load()['data']

    generate_dataset_wav(path)

    train_test_split_wav()