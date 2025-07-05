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
#### Python Dependencies
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

We provide one of the smallest datasets, PEMS08, as a demo, which is located in the `PEMS_Covid19/datasets` folder. Enter the `PEMS_Covid19` directory and run `demo.sh`. 

* Linux

  * ````shell
    sh demo.sh
    ````

* Windows

  * ```shell
    python train.py -c baselines/PEMS08.py
    ```

    

We have placed our own training logs in the checkpoints directory as a reference. The estimated training time is 104 minutes.

## Instructions for use

To ensure a fair comparison of methods across different domains, we adopt various experimental frameworks for evaluation.

Traffic Flow:
* For the PEMS series datasets, we use [BasicTS](https://github.com/GestaltCogTeam/BasicTS) as the baseline framework for experiments (see the PEMS_Covid19 folder).
# Dataset Processing Instructions

## PEMS Covid19 Dataset

### Folder Structure
-PEMS Covid19
  -datasets 
    -PEMS08

### Operations
All operations must be performed in the PEMS Covid19 folder after downloading and placing the dataset.

#### Training
python train.py -c baselines/${CONFIG_NAME}.py --gpus '0'
Example:
python train.py -c baselines/PEMS08.py --gpus '0'

#### Testing
python experiments/train.py -c baselines/${CONFIG_NAME}.py --ckpt ${CHECKPOINT_PATH}.pt --gpus '0'
Meteorology:
* We use the original authors' publicly available code [Corrformer](https://github.com/thuml/Corrformer) as the experimental baseline.

Epidemics:
* The raw data originates from [CSSE](https://github.com/CSSEGISandData/COVID-19) and is processed within the BasicTS framework (see the PEMS_Covid19 folder).

Except for the epidemic dataset, the raw and processed data can be accessed through the links above. Additionally, all processed datasets are available at this [link](https://drive.google.com/drive/folders/11xEsQldS-MmVpq8VzIg9HEEhvCUQ7-QV).

After downloading the data, extract the files and run `run.sh` in the respective directories (LargeST/Meteorology/PEMS_Covid19).
