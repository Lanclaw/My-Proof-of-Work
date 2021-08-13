from data_preprocessing.swat_pre import *


if __name__ == '__main__':
    # params
    window_size = 12
    BATCH_SIZE = 7919
    N_EPOCHS = 10
    hidden_size = 100

    model, train_loader, val_loader, test_loader, labels = preprocessing_usad(window_size, BATCH_SIZE, N_EPOCHS, hidden_size)
    model = to_device(model, device)

    checkpoint = torch.load("results/usad.pth")

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])

    results = testing(model, test_loader)

    windows_labels = []
    for i in range(len(labels) - window_size):
        windows_labels.append(list(np.int_(labels[i:i + window_size])))

    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                             results[-1].flatten().detach().cpu().numpy()])
    print('y_pred', y_pred.shape)
    print('result', len(results), results[0].shape)

    threshold = ROC(y_test, y_pred)
