'''
Dataset preprocess for CUHK02
'''

import os
from re import L
from shutil import copyfile

print("*****************************************************")
print("Currently processing CUHK02")
print("*****************************************************")

download_path = f'/home/b07611033/decentralized_reproduce/datasets/cuhk02'
save_path = download_path + '/pytorch'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

datatype = 'png'

datasets = {'cuhk03-np-detected':1467, 'DukeMTMC-reID':7140, 'Market':1500, 'MSMT17':1040}
#datasets_last_id = [1467, 7140, 1500, 1040]
cuhk02_cam_pair = {'P1':0, 'P2':971, 'P3':306, 'P4':107, 'P5':193}
#cuhk02_cam_pair = {'P1':971, 'P2':306, 'P3':107, 'P4':193, 'P5':239}
pair = ['cam1', 'cam2']
for dataset in datasets.keys():
    print("*****************************************************")
    print(f"Copying files from CUHK02 to {dataset}")
    print("*****************************************************")
    train_save_path = f'/home/b07611033/decentralized_reproduce/datasets/{dataset}/pytorch/train'
    val_save_path = f'/home/b07611033/decentralized_reproduce/datasets/{dataset}/pytorch/val'
    current_id = datasets[dataset]
    for cam_pair in cuhk02_cam_pair: # P1 to P5
        current_id = current_id + cuhk02_cam_pair[cam_pair]
        print(current_id)
        for cam in pair: # cam1 or cam2
            if cam == 'cam1':
                c = 'c1'
            else:
                c = 'c2'
            train_path = download_path + '/' + cam_pair + '/' + cam
            if not os.path.isdir(train_save_path):
                os.mkdir(train_save_path)
            for root, dirs, files in os.walk(train_path, topdown = True):
                for name in files:
                    if not name[-3:] == datatype:
                        continue
                    id = name.split('_')
                    src_path = train_path + '/' + name
                    new_id = str(int(id[0]) + current_id)
                    dst_path = train_save_path + '/' + new_id
                    new_name = new_id + '_' + c + '_' + 'cuhk02.png'
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                        dst_path = val_save_path + '/' + new_id
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + new_name)
                    
print("*****************************************************")
print("Training set and validation set processing done")
print("*****************************************************")


