import numpy as np
import torch

serial_num_path = 'data/number.txt'
well_num = 132 # 井个数


def get_adj(graph_path):
    graph_file = open(graph_path, 'r', encoding='utf-8')
    adj_mat = np.zeros((well_num, well_num))

    for line in graph_file.readlines():
        x = int(line.split('\t')[0])
        y = int(line.split('\t')[1])
        adj_mat[x - 1][y - 1] = 1

    return adj_mat


def load_dataset(graph_name, datadir, dataset):
    adj_mat = get_adj('data/graph.cites.' + graph_name)

    print('Loading {} dataset...'.format(dataset))
    idx_features = np.genfromtxt("{}{}.content".format(datadir, dataset), dtype=np.float32)
    features = idx_features[:, 1:]

    features_len = int(features.shape[0] / 132)
    interval = 0
    i = 0
    X = None
    while i < 132:
        xt = np.expand_dims(features[interval * features_len:(interval + 1) * features_len, :], axis=0)
        interval += 1
        if X is None:
            X = xt
        else:
            X = np.concatenate((X, xt), axis=0)
        i += 1
    X = X.transpose(0, 2, 1) # X.shape = (132, 1, 945)

    return adj_mat, X


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
                in range(X.shape[2] - (
                    num_timesteps_input + num_timesteps_output) + 1)]
    features, target = [], []
    # TODO: 这里有点没太理解，这里的features和target究竟是什么？？有什么关联？
    for i, j in indices:
        features.append(X[:, :, i: i + num_timesteps_input].transpose(0, 2, 1))
        target.append(X[:, -1, i + num_timesteps_input:j])
    
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))


def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                        diag.reshape((1, -1)))
    return A_wave