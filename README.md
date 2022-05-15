# pytorch implementation for "Anomalous Sound Detection Using Spectral-Temporal Information Fusion"
https://ieeexplore.ieee.org/document/9747868.
### Installation

---

```shell
$ conda create -n stgram_mfn python=3.7
$ conda activate stgram_mfn
$ pip install -r requirements.txt
$ python run.py
```

### dataset
We manually mixed the development and additional training dataset of DCASE 2020 Task2
+ development dataset: https://zenodo.org/record/3678171
+ additional training dataset: https://zenodo.org/record/3727685
If you want test the result in evaluation dataset (https://zenodo.org/record/3841772#.YoCDkOhBxaQ), you can use official evaluator: https://github.com/y-kawagu/dcase2020_task2_evaluator
data directory tree:
```text
data
├── dataset
│   ├── fan
│   │   ├── test
│   │   └── train
│   ├── pump
│   │   ├── test
│   │   └── train
│   ├── slider
│   │   ├── test
│   │   └── train
│   ├── ToyCar
│   │   ├── test
│   │   └── train
│   ├── ToyConveyor
│   │   ├── test
│   │   └── train
│   └── valve
│       ├── test
│       └── train
├── pre_data
│   └── 313frames_train_path_list.db
```

### Result
 | Machine Type | AUC(%) | pAUC(%) | mAUC(%) |
 | --------     | :-----:| :----:  | :----:  |
 | Fan          | 94.04  | 88.97   | 81.39   |
 | Pump         | 91.94  | 81.75   | 83.48   |
 | Slider       | 99.55  | 97.61   | 98.22   |
 | Valve        | 99.64  | 98.44   | 98.83   |
 | ToyCar       | 94.44  | 87.68   | 83.07   |
 | ToyConveyor  | 74.57  | 63.60   | 64.16   |
 | Average      | 92.36  | 86.34   | 84.86   |
 
 ### Cite
 If you think this work is useful to you, please cite:
 ```text
@INPROCEEDINGS{9747868,
  author={Liu, Youde and Guan, Jian and Zhu, Qiaoxi and Wang, Wenwu},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Anomalous Sound Detection Using Spectral-Temporal Information Fusion}, 
  year={2022},
  volume={},
  number={},
  pages={816-820},
  doi={10.1109/ICASSP43922.2022.9747868}}
```
