# pytorch implementation for "Anomalous Sound Detection Using Spectral-Temporal Information Fusion"
paper link: https://arxiv.org/pdf/2201.05510.pdf, it had been accepted by ICASSP 2022.
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