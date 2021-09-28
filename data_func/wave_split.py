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
    wav_length = (frames-1) * hop_length + win_length
    skip_length = skip_frames * hop_length
    # wav_vector = np.zeros(((y.shape[0]-wav_length)//skip_length, wav_length))
    for i in range((y.shape[0]-wav_length)//skip_length):
        wav_vector = y[i*skip_length: i*skip_length+wav_length]
        split_path = os.path.join(save_data_dir, f'{machine}_{wav_name}_{i}.wav')
        split_path_list.append(split_path)
        # print(spilit_path)
        sf.write(split_path, data=wav_vector, samplerate=sr)

    return split_path_list

# def file_list_split(file_list,
#                     save_data_dir,
#                     n_fft=1024,
#                     hop_length=512,
#                     frames=5,
#                     skip_frames=1):
#     path_list = []
#     for file in file_list:
#         paths = file_split(file, save_data_dir,
#                            n_fft=n_fft,
#                            hop_length=hop_length,
#                            frames=frames,
#                            skip_frames=skip_frames)
#         path_list.append(paths)
#     return path_list


# def data_split(process_machines,
#                data_dir,
#                save_dir,
#                dir_name='train',
#                domain='source',
#                data_type='dev_data',
#                thread_num=10,
#                win_length=1024,
#                hop_length=512,
#                frames=5,
#                skip_frames=1,):
#     dirs = utils.select_dirs(data_dir, data_type=data_type)
# 
#     for index, target_dir in enumerate(sorted(dirs)):
#         print('\n' + '=' * 20)
#         print(f'[{index + 1}/{len(dirs)}] {target_dir}')
#         time.sleep(1)
#         machine_type = os.path.split(target_dir)[1]
# 
#         path_list = []
#         path_list_path = os.path.join(save_dir, data_type, f'{machine_type}_{dir_name}_{domain}_path_list.db')
# 
#         if machine_type not in process_machines:
#             continue
#         save_data_dir = os.path.join(save_dir, data_type, dir_name, machine_type)
#         # print(save_dir, save_data_dir)
#         os.makedirs(save_data_dir, exist_ok=True)
# 
#         # ms_file_path = os.path.join(pre_data_dir, machine_type, f'{machine_type}_mean_std.db')
#         # with open(ms_file_path, 'rb') as f:
#         #     ms_data = joblib.load(f)
#         # mean, std = ms_data['mean'], ms_data['std']
# 
#         files = utils.create_file_list(target_dir,
#                                        domain=domain,
#                                        dir_name=dir_name,
#                                        ext='wav')
# 
#         pool = mul.Pool(thread_num)
#         partial_work = partial(file_split, save_data_dir=save_data_dir,
#                                            win_length=win_length,
#                                            hop_length=hop_length,
#                                            frames=frames,
#                                            skip_frames=skip_frames)
#         path_list.append(pool.map(partial_work, files))
# 
#         path_list = list(chain.from_iterable(chain.from_iterable(path_list)))
#         print(f'{data_type} {domain} domain {machine_type} were split to {len(path_list)} wav files!')
#         with open(path_list_path, 'wb') as f:
#             joblib.dump(path_list, f)

def data_split(process_machines,
               data_dir,
               save_dir,
               dir_name='train',
               data_type=''):
    dirs = utils.select_dirs(data_dir, data_type=data_type)
    path_list_path = os.path.join(save_dir, data_type, f'313frames_{dir_name}_path_list.db')
    path_list = []
    for index, target_dir in enumerate(sorted(dirs)):
        print('\n' + '=' * 20)
        print(f'[{index + 1}/{len(dirs)}] {target_dir}')
        time.sleep(1)
        machine_type = os.path.split(target_dir)[1]
        if machine_type not in process_machines:
            continue
            
        files = utils.create_file_list(target_dir,
                                       dir_name=dir_name,
                                       ext='wav')
        path_list.append(files)
        
        print(f'{data_type} {machine_type} were split to {len(files)} wav files!')
    path_list = list(chain.from_iterable(path_list))
    with open(path_list_path, 'wb') as f:
        joblib.dump(path_list, f)