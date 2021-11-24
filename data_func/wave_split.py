import librosa
import soundfile as sf
import numpy as np
import os
import time
import tqdm
import threading
from concurrent import futures
import joblib
import multiprocessing as mul
from functools import partial
from itertools import chain

import utils


def file_split(file_name,
               save_data_dir,
               win_length=1024,
               hop_length=512,
               frames=5,
               skip_frames=1):
    path_list = file_name.split('/')
    split_path_list = []
    machine, wav_name = path_list[-3], path_list[-1]

    y, sr = librosa.load(file_name, sr=None)
    wav_length = (frames - 1) * hop_length + win_length
    skip_length = skip_frames * hop_length
    # wav_vector = np.zeros(((y.shape[0]-wav_length)//skip_length, wav_length))
    for i in range((y.shape[0] - wav_length) // skip_length):
        wav_vector = y[i * skip_length: i * skip_length + wav_length]
        split_path = os.path.join(save_data_dir, f'{machine}_{wav_name}_{i}.wav')
        split_path_list.append(split_path)
        # print(spilit_path)
        sf.write(split_path, data=wav_vector, samplerate=sr)

    return split_path_list


# ================
# machine types
# ================
# def data_split(process_machines,
#                data_dir,
#                save_dir,
#                dir_name='train',
#                data_type=''):
#     dirs = utils.select_dirs(data_dir, data_type=data_type)
#     path_list_path = os.path.join(save_dir, data_type, f'313frames_{dir_name}_path_list_dict.db')
#     path_list_dict = {}
#     for index, target_dir in enumerate(sorted(dirs)):
#         print('\n' + '=' * 20)
#         print(f'[{index + 1}/{len(dirs)}] {target_dir}')
#         time.sleep(1)
#         machine_type = os.path.split(target_dir)[1]
#         if machine_type not in process_machines:
#             continue
#
#         files = utils.create_file_list(target_dir,
#                                        dir_name=dir_name,
#                                        ext='wav')
#         path_list_dict[machine_type] = files
#
#         print(f'{data_type} {machine_type} were split to {len(files)} wav files!')
#     # path_list = list(chain.from_iterable(path_list))
#     with open(path_list_path, 'wb') as f:
#         joblib.dump(path_list_dict, f)

def data_split(process_machines,
               data_dir,
               root_folder,
               ID_factor,
               dir_name='train',
               data_type='',):
    dirs = utils.select_dirs(data_dir, data_type=data_type)
    path_list_dict = {}
    for index, target_dir in enumerate(sorted(dirs)):
        print('\n' + '=' * 20)
        print(f'[{index + 1}/{len(dirs)}] {target_dir}')
        time.sleep(1)
        machine_type = os.path.split(target_dir)[1]
        if machine_type not in process_machines:
            continue

        machine_id_list = utils.get_machine_id_list(target_dir, dir_name=dir_name)
        for id_str in machine_id_list:
            files, _ = utils.create_test_file_list(target_dir, id_str, dir_name=dir_name)
            if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                id = int(id_str[-1]) - 1
            else:
                id = int(id_str[-1])

            path_list_dict[ID_factor[machine_type]*7 + id] = files

            print(f'{data_type} {machine_type} {id_str} were split to {len(files)} wav files!')
    # path_list = list(chain.from_iterable(path_list))
    with open(root_folder, 'wb') as f:
        joblib.dump(path_list_dict, f)
