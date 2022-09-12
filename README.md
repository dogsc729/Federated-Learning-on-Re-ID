# Federated-Learning-on-Re-ID

### Abstract

This project covers experiments of my research about Federated Learning on Person Re-Identification. Our primary goal is to jointly optimize performance on seen domain and unseen domain.

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
5. Start the training by `python3 ./src/federated_train.py`