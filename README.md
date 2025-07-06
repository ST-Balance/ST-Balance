#  ST-Balance
## Overview
Accurate spatiotemporal prediction is critical in fields such as urban traffic management, meteorology, and public health monitoring. However, existing methods face significant performance bottlenecks, typically delivering only incremental improvements and often lacking universal applicability across different domains. A key reason for these limitations is an imbalance between spatial complexity and temporal constraints, quantified here through spatial and temporal entropy measures. This quantification reveals that spatial complexity frequently exceeds temporal modeling capacities, increasing prediction uncertainty. To address these fundamental challenges, this study introduces a scalable, adaptive framework that harmonizes spatiotemporal feature dimensions with linear computational complexity. Spatial dimensionality is reduced using low-rank matrix decomposition, preserving essential structural information, while temporal horizons are extended to capture long-term dependencies, mitigating cumulative errors induced by spatial heterogeneity. Extensive experiments across diverse datasets—including urban traffic flow, meteorological forecasting, and epidemic scenarios—demonstrate substantial improvements in predictive accuracy and robust cross-domain generalization. This work thus provides a foundational advancement in balancing spatial and temporal complexity, enabling accurate and efficient prediction across diverse spatiotemporal applications.
## System Requirements
### Hardware Requirements
All experiments were conducted on a computational server equipped with an Intel(R) Core(TM) i7-9700K CPU @ 3.60 GHz and an NVIDIA RTX A6000 GPU with 48GB of memory. 
### Software Requirements
#### OS Requirements
Our experiments are all running in the system:
* Linux: Ubuntu 18
#### Python Version
Python 3.9 or higher is required
#### CUDA VERSION
our cuda version is 11.4
#### PyTorch Dependencies
````
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
````
#### Python Dependencies
After ensuring PyTorch is installed correctly, you can install the other dependencies:
````
pip install -r requirements.txt
````
ST-Balance mainly depends on the Python scientific stack.
````
easy-torch==1.3.2
easydict==1.10
pandas==1.3.5
packaging==23.1
setuptools==59.5.0
scipy==1.7.3
tables==3.7.0
sympy==1.10.1
setproctitle==1.3.2
scikit-learn==1.0.2
einops==0.6.1
matplotlib==3.8.2
numpy==1.22.4
networkx==2.6.3
pyyaml==6.0.1
karateclub==1.3.3
node2vec==0.4.6
umap-learn==0.5.3
````
## Installation Guide
````
git clone https://github.com/ST-Balance/ST-Balance.git 
````
* It takes a few seconds.
## Demo

We provide one of the smallest datasets, PEMS08, as a demo, which is located in the `PEMS/datasets` folder. Enter the `PEMS` directory and run `demo.sh`. Other datasets need to be decompressed or downloaded before they can be used.

* Linux

  * ````shell
    sh demo.sh
    ````

* Windows

  * ```shell
    python train.py -c config/PEMS08.py
    ```

    

We have placed our own training logs in the checkpoints directory as a reference. The estimated training time is 104 minutes.

## Instructions for use

To ensure a fair comparison of methods across different domains, we adopt various experimental frameworks for evaluation.

Traffic Flow:
* For the PEMS series datasets, we use [BasicTS](https://github.com/GestaltCogTeam/BasicTS/tree/master/baselines) as the baseline framework for experiments (see the PEMS folder).
* For the LargeST series datasets, we adopt the original authors' publicly available code [LargeST](https://github.com/liuxu77/LargeST/tree/main/experiments) as the experimental baseline (see the LargeST folder).

Meteorology:
* We use the original authors' publicly available code [Corrformer](https://github.com/thuml/Corrformer) as the experimental baseline.

Epidemics:
* The raw data originates from [CSSE](https://github.com/CSSEGISandData/COVID-19) and is processed within the BasicTS framework (see the Covid19 folder). We also use the [STSGT](https://github.com/soumbane/STSGT), [SAB‑GNN](https://github.com/JiaweiXue/MultiwaveCovidPrediction) and [EpiLearn](https://github.com/Emory-Melody/EpiLearn/tree/main/epilearn/models/SpatialTemporal) as the experimental baseline.

## Dataset
Except for the epidemic dataset, the raw and processed data can be accessed through the links above. Additionally, all processed datasets are available at this [link](https://drive.google.com/drive/folders/11xEsQldS-MmVpq8VzIg9HEEhvCUQ7-QV).

## FOR PEMS Dataset
### Folder Structure
````shell
-PEMS
  -datasets 
    -PEMS08
````
### Operations
Once the dataset has been downloaded and placed in the PEMS folder, all operations should be conducted within this folder. [DataLink](https://drive.google.com/file/d/1mm-Yc3lIsUue1DBWvgOtuSBXOHOGur5g/view?usp=drive_link) 

#### Training
````shell
python train.py -c config/${CONFIG_NAME}.py --gpus '0'
````
Example:
````shell
python train.py -c config/PEMS08.py --gpus '0'
````
#### Testing
````shell
python experiments/train.py -c config/${CONFIG_NAME}.py --ckpt ${CHECKPOINT_PATH}.pt --gpus '0'
````
## For LargeST Dataset

### Folder Structure
````shell
-LargeST
  -data
    -sd
````
### Operations
Once the dataset has been downloaded and placed in the LargeST folder, all operations should be conducted within this folder. [DataLink](https://drive.google.com/file/d/149Qs98mx9sg9lIPBjXtf6iERPPqp9Zf2/view?usp=drive_link) 

#### Training
````shell
python main.py --dataset ${DATA_SET_NAME} --mode 'train'
````
Example:
````shell
python main.py --dataset SD --mode 'train'
````
Note: Parameter configurations can be viewed in main.py and config/ST-Balance.yaml. Hyperparameter configurations are available in src/utils/args.py.

#### Testing
````shell
python main.py --dataset ${DATA_SET_NAME} --mode 'test'
````
Example:
````shell
python main.py --dataset SD --mode 'test'
````
## For Meteorology Dataset

### Folder Structure
````shell
-Meteorology
  -dataset
  -adj_ang.npy
````
### Operations
Once the dataset has been downloaded and placed in the Meteorology folder, all operations should be conducted within this folder. [DataLink](https://drive.google.com/file/d/1TG8VQGuvhGErIkNUU7JYvJIULtiLwWGs/view?usp=drive_link) 

#### Training
````shell
python run.py --is_training 1 -root_path ./dataset/global_wind/ --data Global_Win
````
Note: Parameter configurations can be viewed in run.py.

#### Testing
````shell
python run.py --is_training 0 -root_path ./dataset/global_wind/ --data Global_Win
````
## FOR Covid19 Dataset
### Folder Structure
````shell
-Covid19
  -datasets 
    -Covid19_US
````
### Operations
Once the dataset has been downloaded and placed in the Covid19 folder, all operations should be conducted within this folder. [DataLink](https://drive.google.com/file/d/16PGCd2C4tgU5PbMeQXOSITx5cm-gRkRd/view?usp=drive_link) 

#### Training
````shell
python train.py -c config/${CONFIG_NAME}.py --gpus '0'
````
Example:
````shell
python train.py -c config/Covid19_US.py --gpus '0'
````
#### Testing
````shell
python experiments/train.py -c config/${CONFIG_NAME}.py --ckpt ${CHECKPOINT_PATH}.pt --gpus '0'
````
