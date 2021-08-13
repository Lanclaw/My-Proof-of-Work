from data_preprocessing.swat_pre import *
from model.USAD import *


if __name__ == '__main__':
    # data params
    BATCH_SIZE = 7919
    N_EPOCHS = 100
    # model params
    window_size = 12
    hidden_size = 100

    train_loader, val_loader, test_loader, labels, features_num = preprocessing_swat(window_size, BATCH_SIZE, N_EPOCHS, hidden_size)
    w_size = window_size * features_num
    z_size = window_size * hidden_size
    model = UsadModel(w_size, z_size)
    model = to_device(model, device)

    history = training(N_EPOCHS, model, train_loader, val_loader)
    plot_history(history)

    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder1': model.decoder1.state_dict(),
        'decoder2': model.decoder2.state_dict()
    }, "results/usad.pth")

