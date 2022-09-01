'''
Dataset Preprocess
'''

import os
from shutil import copyfile

dataset_list = ['cuhk03-np-detected', 'DukeMTMC-reID', 'Market', 'MSMT17']

for dataset in dataset_list:

    print("*****************************************************")
    print(f"Currently processing {dataset}")
    print("*****************************************************")

    download_path = f'/home/b07611033/decentralized_reproduce/datasets/{dataset}'
    save_path = download_path + '/pytorch'

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if dataset == 'cuhk03-np-detected':
        datatype = 'png'
    else:
        datatype = 'jpg'
    '''
    Train_val
    '''

    train_path = download_path + '/bounding_box_train'
    train_save_path = download_path + '/pytorch/train'
    val_save_path = download_path + '/pytorch/val'

    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
    if not os.path.isdir(val_save_path):
        os.mkdir(val_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:] ==datatype:
                continue
            id = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + id[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = val_save_path + '/' + id[0]
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

    print("*****************************************************")
    print("Training set and validation set processing done")
    print("*****************************************************")

    '''
    Query
    '''

    query_path = download_path + '/query'
    query_save_path = download_path + '/pytorch/query'

    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:] == datatype:
                continue
            id = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + id[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

    print("*****************************************************")
    print("Query processing done")
    print("*****************************************************")

    '''
    Gallery
    '''

    gallery_path = download_path + '/bounding_box_test'
    gallery_save_path = download_path + '/pytorch/gallery'

    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-3:] == datatype:
                continue
            id = name.split('_')
            src_path = gallery_path + '/' + name
            dst_path = gallery_save_path + '/' + id[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

    print("*****************************************************")
    print("Gallery processing done")
    print("*****************************************************")
