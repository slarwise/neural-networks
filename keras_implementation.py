import numpy as np
import pickle
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, metrics, regularizers
from keras.callbacks.callbacks import EarlyStopping

def train_network(network_number):
    x_train, t_train, x_valid, t_valid, x_test, t_test = load_CIFAR()

    # Center input data
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_valid -= x_train_mean
    x_test -= x_train_mean

    model = get_model(network_number, x_train.shape[1])

    opt = get_optimizer(network_number)
    
    model.compile(loss="categorical_crossentropy", optimizer=opt)

    early_stopping = EarlyStopping(monitor="val_loss", patience=3)

    results = model.fit(x=x_train, y=t_train, batch_size=8192, epochs=400,
            shuffle=True, validation_data=(x_valid, t_valid),
            validation_freq=30, callbacks=[early_stopping],
            )

    n_epochs = len(results.history["loss"])

    pred_train = model.predict(x=x_train)
    pred_valid = model.predict(x=x_valid)
    pred_test = model.predict(x=x_test)

    return (classification_error(pred_train, t_train),
            classification_error(pred_valid, t_valid),
            classification_error(pred_test, t_test),
            n_epochs)

def get_model(network_number, input_shape):
    if network_number == 1:
        return Sequential([
            Dense(50, input_shape=(input_shape,)),
            Activation("relu"),
            Dense(50),
            Activation("relu"),
            Dense(10),
            Activation("softmax"),
            ])
    if network_number == 2:
        return Sequential([
            Dense(50, input_shape=(input_shape,)),
            Activation("relu"),
            Dense(50),
            Activation("relu"),
            Dense(50),
            Activation("relu"),
            Dense(10),
            Activation("softmax"),
            ])
    if network_number == 3:
        l2_par = 0.2
        return Sequential([
            Dense(50, input_shape=(input_shape,),
                kernel_regularizer=regularizers.l2(l2_par)),
            Activation("relu"),
            Dense(50, kernel_regularizer=regularizers.l2(l2_par)),
            Activation("relu"),
            Dense(10, kernel_regularizer=regularizers.l2(l2_par)),
            Activation("softmax"),
            ])

def get_optimizer(network_number):
    if network_number in (1, 3):
        return optimizers.SGD(learning_rate=0.001, momentum=0.9)
    if network_number == 2:
        return optimizers.SGD(learning_rate=0.003, momentum=0.9)

def classification_error(o, t):
    p = o.shape[0]
    y = np.zeros_like(o)
    y[np.arange(p), np.argmax(o, axis=1)] = 1
    return 1/(2*p) * np.sum(np.absolute(t-y))

def load_CIFAR():
    x_valid = np.array([])
    t_valid = np.array([])
    x_test = np.array([])
    t_test = np.array([])

    directory = "cifar-10-batches-py/"
    x_train = np.zeros((40000, 3072))
    t_train = np.zeros((40000, 10), dtype="int")
    start_row = 0
    for filename in [directory + "data_batch_" + str(i) for i in range(1, 5)]:
        train_dict = unpickle(filename)
        x_train[start_row:start_row+10000, :] = train_dict.get(b'data')
        t_train[list(range(start_row, start_row+10000)),
                train_dict.get(b'labels')] = 1
        start_row += 10000
    x_train /= 255

    val_dict = unpickle(directory + "data_batch_5")
    x_valid = np.zeros((10000, 3072))
    x_valid[:, :] = val_dict.get(b'data')
    x_valid /= 255
    t_valid = np.zeros((10000, 10), dtype="int")
    t_valid[list(range(10000)), val_dict.get(b'labels')] = 1

    test_dict = unpickle(directory + "test_batch")
    x_test = np.zeros((10000, 3072))
    x_test[:, :] = test_dict.get(b'data')
    x_test /= 255
    t_test = np.zeros((10000, 10), dtype="int")
    t_test[list(range(10000)), test_dict.get(b'labels')] = 1

    return x_train, t_train, x_valid, t_valid, x_test, t_test

def unpickle(filename):
    with open(filename, "rb") as f:
        dict = pickle.load(f, encoding="bytes")
    return dict

if __name__ == "__main__":
    c_error_train = np.zeros(3)
    c_error_valid = np.zeros(3)
    c_error_test = np.zeros(3)
    network_numbers = np.array([1, 2, 3])
    for i, net in enumerate(network_numbers):
        c_error_train[i], c_error_valid[i], c_error_test[i], n_epochs = \
                train_network(net)
        
    df = pd.DataFrame({"Network": network_numbers,
        "Epochs": n_epochs,
        "Training": c_error_train,
        "Validation": c_error_valid,
        "Test": c_error_test})
    df.to_csv("../data/q3_class_errors.csv", index=False)
