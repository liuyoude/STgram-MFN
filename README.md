## pytorch implementation for "Anomalous Sound Detection Using Spectral-Temporal Information Fusion"
You can find paper in this link: https://ieeexplore.ieee.org/document/9747868.
![structure](./structure.png)
### Installation

---

```shell
$ conda create -n stgram_mfn python=3.7
$ conda activate stgram_mfn
$ pip install -r requirements.txt
$ python -m visdom.server
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
└── pre_data
    └── 313frames_train_path_list.db  # Automatically generated file
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
 
 ```text
ToyCar		
id	AUC	pAUC
1	0.830719697	0.652198679
2	0.951617251	0.874308413
3	0.995218329	0.981046957
4	0.99993531	0.999659526
Average	0.944372647	0.876803394
ToyConveyor		
id	AUC	pAUC
1	0.869696875	0.766776316
2	0.641580986	0.547164566
3	0.725856264	0.594070052
Average	0.745711375	0.636003645
fan		
id	AUC	pAUC
0	0.950638821	0.894607526
2	0.996852368	0.983433514
4	0.813936782	0.681034483
6	0.999972299	0.999854206
Average	0.940350067	0.889732432
pump		
id	AUC	pAUC
0	0.892342657	0.744939271
2	0.83481982	0.693219535
4	0.9999	0.999473684
6	0.950490196	0.832301342
Average	0.919388168	0.817483458
slider		
id	AUC	pAUC
0	1	1
2	0.982209738	0.906367041
4	0.99988764	0.999408634
6	0.999775281	0.998817268
Average	0.995468165	0.976148236
valve		
id	AUC	pAUC
0	1	1
2	0.988333333	0.952192982
4	1	1
6	0.99725	0.985526316
Average	0.996395833	0.984429825
Total Average	0.923614376	0.863433498
```
 
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
