#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
import numpy as np
import pickle
import pandas as pd

def train_network(network_number):
    x_train, t_train, x_valid, t_valid, x_test, t_test = load_CIFAR()

    # Center input data
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_valid -= x_train_mean
    x_test -= x_train_mean

    # Initialize network
    nu = 0.1
    n_epochs = 20
    m_B = 100

    # Decide layout, M[i] is the number of neurons in layer i
    layout = {
            1: [x_train.shape[1], 10],
            2: [x_train.shape[1], 10, 10],
            3: [x_train.shape[1], 50, 10],
            4: [x_train.shape[1], 50, 50, 10]
            }
    M = layout[network_number]
    print("Network {}".format(network_number))
    L = len(M) - 1

    V_batch = []
    V_train = []
    V_valid = []
    V_test = []
    for l in range(L+1):
        V_batch.append(np.zeros((m_B, M[l])))
        V_train.append(np.zeros((x_train.shape[0], M[l])))
        V_valid.append(np.zeros((x_valid.shape[0], M[l])))
        V_test.append(np.zeros((x_test.shape[0], M[l])))
    W = [[]]
    theta = [[]]
    error_batch = [[]]
    for l in range(1, L+1):
        W.append(np.random.normal(loc=0, scale=1/np.sqrt(M[l-1]),
            size=(M[l], M[l-1])))
        theta.append(np.zeros(M[l]))
        error_batch.append(np.zeros((m_B, M[l])))

    C_train = np.zeros(n_epochs)
    C_valid = np.zeros(n_epochs)
    best_C_valid = np.inf
    best_epoch = None
    best_W = None
    best_theta = None

    for epoch in range(n_epochs):
        print("--------------")
        print("Epoch {}:".format(epoch))

        # Shuffle x_train and t_train
        rand_perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[rand_perm, :]
        t_train = t_train[rand_perm, :]

        # Perform stochastich gradient descent on mini-batches
        for start_index in range(0, x_train.shape[0], m_B):
            # Propagate forward
            V_batch[0][:, :] = x_train[start_index:start_index+m_B, :]
            for l in range(1, L+1):
                V_batch[l][:, :] = compute_current_layer(W[l],
                        V_batch[l-1], theta[l])

            # Propagate backward
            targets = t_train[start_index:start_index+m_B, :]
            b = compute_current_b(W[L], V_batch[L-1], theta[L])
            error_batch[L][:, :] = compute_output_error(b, V_batch[L], targets)
            for l in range(L, 1, -1):
                b = compute_current_b(W[l-1], V_batch[l-2], theta[l-1])
                error_batch[l-1][:, :] = compute_current_error(error_batch[l],
                        W[l], b)

            # Update weights and thresholds
            for l in range(1, L+1):
                W[l][:, :] += compute_weight_increment(error_batch[l],
                        V_batch[l-1], nu)
                theta[l][:] += compute_theta_increment(error_batch[l], nu)

        # Classify train and validations sets
        V_train[0][:, :] = x_train
        V_valid[0][:, :] = x_valid
        for l in range(1, L+1):
            V_train[l][:, :] = compute_current_layer(W[l], V_train[l-1],
                    theta[l])
            V_valid[l][:, :] = compute_current_layer(W[l], V_valid[l-1],
                    theta[l])

        #Compute the classification error
        C_train[epoch] = classification_error(V_train[L], t_train)
        C_valid[epoch] = classification_error(V_valid[L], t_valid)

        # Save weights if classification error lower than previous best
        if C_valid[epoch] < best_C_valid:
            best_C_valid = C_valid[epoch]
            best_epoch = epoch
            best_W = W.copy()
            best_theta = theta.copy()

        print("C_train[epoch]: {}".format(C_train[epoch]))
        print("C_valid[epoch]: {}".format(C_valid[epoch]))
        print("Best C_valid: {}".format(best_C_valid))

    # Compute the classification error on the test set
    V_test[0][:, :] = x_test
    for l in range(1, L+1):
        V_test[l][:, :] = compute_current_layer(best_W[l], V_test[l-1],
                best_theta[l])
    C_test = classification_error(V_test[L], t_test)

    # Show the classification errors for the best epoch
    print("best_epoch {}".format(best_epoch))
    print("C_valid[best_epoch]: {}".format(C_valid[best_epoch]))
    print("C_train[best_epoch]: {}".format(best_C_valid))
    print("C_test[best_epoch]: {}".format(C_test))

    # Save results
    class_errors_df = pd.DataFrame({"Network": network_number,
        "Epoch": np.arange(n_epochs),
        "C_train": C_train,
        "C_valid": C_valid})
    class_errors_table_df = pd.DataFrame({"Network": network_number,
        "Epoch": best_epoch,
        "C_train": C_train[best_epoch],
        "C_valid": C_valid[best_epoch],
        "C_test": C_test},
        index=[0])
    return class_errors_df, class_errors_table_df

def compute_current_b(W_current, V_prev, theta_current):
    return np.einsum("ik,jk->ji", W_current, V_prev) - theta_current

def compute_current_layer(W_current, V_prev, theta_current):
    b_current = compute_current_b(W_current, V_prev, theta_current)
    return sigmoid(b_current)

def compute_output_error(b_current, output, t):
    return sigmoid_prime(b_current) * (t - output)

def compute_current_error(error_next, W_next, b_current):
    return np.einsum("ik,kj->ij", error_next, W_next) * sigmoid_prime(b_current)

def compute_weight_increment(error_current, V_prev, nu):
    return nu * np.einsum("ki,kj->ij", error_current, V_prev)

def compute_theta_increment(error_current, nu):
    return -nu * np.einsum("ki->i", error_current)

def classification_error(o, t):
    p = o.shape[0]
    y = np.zeros_like(o)
    y[np.arange(p), np.argmax(o, axis=1)] = 1
    return 1/(2*p) * np.sum(np.absolute(t-y))

def sigmoid(b):
    return 1 / (1+np.exp(-b))

def sigmoid_prime(b):
    sig = sigmoid(b)
    return sig * (1-sig)

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
    class_errors_df, class_errors_table_df = train_network(1)
    for network_number in range(2, 5):
        class_errors_df_i, class_errors_table_df_i = \
                train_network(network_number)
        class_errors_df = class_errors_df.append(class_errors_df_i)
        class_errors_table_df = class_errors_table_df.append(
                class_errors_table_df_i, ignore_index=True)
    # Save to file
    class_errors_df.to_csv("../data/class_errors.csv", index=False)
    class_errors_table_df.to_csv("../data/class_errors_table.csv", index=False)

