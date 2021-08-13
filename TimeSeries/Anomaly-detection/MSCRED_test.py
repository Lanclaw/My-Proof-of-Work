import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
from model.MSCRED import MSCRED
from data_preprocessing.MSCRED_pre import load_data
import matplotlib.pyplot as plt
import numpy as np
import os


def test(dataLoader, model):
    print("------Testing-------")
    index = 800
    loss_list = []
    reconstructed_data_path = "./datasets/Synthetic_matrix_data/reconstructed_data/"
    with torch.no_grad():
        for x in dataLoader:
            x = x.to(device)
            x = x.squeeze()
            reconstructed_matrix = model(x)
            path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(index) + ".npy")
            np.save(path_temp, reconstructed_matrix.cpu().detach().numpy())
            # l = criterion(reconstructed_matrix, x[-1].unsqueeze(0)).mean()
            # loss_list.append(l)
            # print("[test_index %d] [loss: %f]" % (index, l.item()))
            index += 1


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    dataLoader = load_data()
    mscred = MSCRED(3, 256)

    mscred.load_state_dict(torch.load("./results/mscred.pth"))
    mscred.to(device)
    test(dataLoader["test"], mscred)

