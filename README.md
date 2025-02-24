# BTCDMNet: Bayesian Tile Attention Network for Hyperspectral Mangroves Change Detection

***
# Introduction

<b> Official implementation of BTCDMNet by [Junshen Luo](https://github.com/JeasunLok), Jiahe Li, Xinlin Chu, Sai Yang, Lingjun Tao and Qian Shi. </b>
***

***
## How to use it?
### 1. Installation
```
git clone https://github.com/JeasunLok/BTCDMNet.git && cd BTCDMNet
conda create -n BTCDMNet python=3.9
conda activate BTCDMNet
pip install -r requirements.txt
```

### 2. Download our datasets

Download our datasets then place them in `data` folder

Baiduyun: https://pan.baidu.com/s/1hyye2fVxoUaOJ6YR_RUSJg 
(access code: js66)

Google Drive: https://drive.google.com/drive/folders/1xe9i95_noh8dBW7sGn_z4uIfePBsa8r2

### 3. Quick start to use our SOTA model BTCDMNet

<b> You should change the settings in config.yaml especially `HSI_data_path` and `model_type` then: </b>
```
chmod +x demo.sh
./demo.sh
```

### 4. More detailed information
see `config.yaml`

***
## Contact Information
Junshen Luo: luojsh7@mail2.sysu.edu.cn

Junshen Luo is with School of Geography and Planning, Sun Yat-sen University, Guangzhou 510275, China
***