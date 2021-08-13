import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
from model.MSCRED import MSCRED
from data_preprocessing.MSCRED_pre import load_data
import matplotlib.pyplot as plt
import numpy as np
import os


def train(dataLoader, model, optimizer, epochs, device):
    model = model.to(device)
    print("------training on {}-------".format(device))
    for epoch in range(epochs):
        train_l_sum, n = 0.0, 0

        for x in tqdm(dataLoader):
            x = x.to(device)
            x = x.squeeze()
            # print(type(x))
            y = model(x)
            l = torch.mean((y - x[-1].unsqueeze(0)) ** 2)
            train_l_sum += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            # print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))

        print("[Epoch %d/%d] [loss: %f]" % (epoch + 1, epochs, train_l_sum / n))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    dataLoader = load_data()
    mscred = MSCRED(3, 256)

    # 训练阶段
    # mscred.load_state_dict(torch.load("./checkpoints/model1.pth"))
    optimizer = torch.optim.Adam(mscred.parameters(), lr = 0.0002)
    train(dataLoader["train"], mscred, optimizer, 10, device)
    print("保存模型中....")
    torch.save(mscred.state_dict(), "./results/mscred.pth")
