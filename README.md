
## Federated Learning on Person Re-Identification
### Abstract

This project covers my research experiments about Federated Learning on Person Re-Identification. Our primary goal is jointly optimizing performance on seen and unseen domains. One feature of this project is reproducing experimental results presented in **Decentralised Person Re-Identification with Selective Knowledge Aggregation**   [1].

### News
* Dec 6th, 2022. IBN-Net [2] is now available. The source code of IBN-Net for ReID is referenced from [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline).

### Source Code

[GitHub Link](https://github.com/dogsc729/Federated-Learning-on-Re-ID)

### Usage

1. Clone the source code by 
```
git clone https://github.com/dogsc729/Federated-Learning-on-Re-ID.git
```

2. Create two directories, **checkpoint** and **datasets**, under the cloned repositories by
```
mkdir checkpoint
```
and
```
mkdir datasets
```

3. You should have four datasets already, including `Market1501`, `MSMT17`, `DukeMTMC-reID` and `CUHK03-np-detected`.

    Unzip the four datasets and place them in `datasets`, the structure should be:
    ```
    datasets
    |- /Market
    |    |- /bounding_box_test
    |    |- /bounding_box_train
    |    |- /gt_bbox
    |    |- /gt_query
    |- /MSMT17
    |    |- /bounding_box_test
    |    |- /bounding_box_train
    |- /DukeMTMC-reID
    |    |- /bounding_box_test
    |    |- /bounding_box_train
    |- /cuhk03-np-detected
        |- /bounding_box_test
        |- /bounding_box_train
    ```
    Note that the structure above including the naming should be exactly the same.
4. Pre-process the datasets by running `python3 ./src/big_data_preprocess.py`
5. Start the training by `python3 ./src/federated_train.py`. In addition, you can change the settings by adding the arguments below.  
   * `-s, --scenario`: You can change the training scenario by selecting `ska` for Selective Knowledge Aggregation or `fed` for classic Federated Learning, The default value is `ska`.
   * `-l, --location`: You can change the location of the directory under `/checkpoint/`. The log file, models, and record of training progress in .png file will be stored here. The default value is the time you start the training.
   * `-m, --model`: You can change the type of model by selecting `attentive` for Attentive normalization ResNet50, `vanilla` for vanilla ResNet50 or `ibn` for IBN-Net. The default value is `attentive`.
   * `--global_iter`: You can change the number of iteration of the global training stage. The default value is `100`.
   * `--local_epoch`: You can change the number of epoch trained on each client model. The default value is `1`.
   * `lr_feature`: You can change the learning rate of the feature extraction layers of the model. The default value is `0.01`.
   * `lr_classifier`: You can change the learning rate of the classifier layers of the model. The default value is `0.1`.  
    
    For example, You can run `python3 ./src/federated_train.py -s fed -l federated_test -m vanilla --local_epoch 5` to set your experiment on classic Federated Learning scenario, checkpoint location at `/checkpoint/federated_test`, using vanilla ResNet50 as your model and set the number of local epoch trained for each global round as `5`.

### Reference

* [1] Shitong Sun, Guile Wu, Shaogang Gong. Decentralised Person Re-Identification with Selective Knowledge Aggregation. In BMVC, 2021.
* [2] Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net, In ECCV, 2018.