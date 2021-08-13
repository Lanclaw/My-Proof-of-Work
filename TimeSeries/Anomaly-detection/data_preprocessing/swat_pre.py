import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

import torch.utils.data as data_utils


def preprocessing_swat(window_size, BATCH_SIZE, N_EPOCHS, hidden_size):

    # Read data
    normal = pd.read_csv("datasets/SWaT_Dataset_Normal_v1.csv")#, nrows=1000)
    normal = normal.drop(["Timestamp" , "Normal/Attack" ], axis=1)
    print('normal1', type(normal))

    # Transform all columns into float64
    for i in list(normal):
        normal[i] = normal[i].apply(lambda x: str(x).replace("," , "."))

    normal = normal.astype(float)
    print('normal2', normal.shape)

    # normalization
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()

    x = normal.values
    x_scaled = min_max_scaler.fit_transform(x)
    normal = pd.DataFrame(x_scaled)

    # attack
    # Read data
    attack = pd.read_csv("datasets/SWaT_Dataset_Attack_v0.csv", sep=";") #, nrows=1000)
    labels = [ float(label != 'Normal') for label in attack["Normal/Attack"].values]
    attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)

    # Transform all columns into float64
    for i in list(attack):
        attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
    attack = attack.astype(float)

    # normalization
    from sklearn import preprocessing

    x = attack.values
    x_scaled = min_max_scaler.transform(x)
    attack = pd.DataFrame(x_scaled)

    # windows

    windows_normal = normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
    windows_attack = attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
    print('windows_normal', windows_normal.shape, windows_attack.shape, window_size)

    features_num = windows_normal.shape[2]
    w_size = window_size * windows_normal.shape[2]
    z_size = window_size * hidden_size

    windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
    ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
    ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
    ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    return train_loader, val_loader, test_loader, labels, features_num

