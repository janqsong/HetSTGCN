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


def get_het_adj(graph_path):
    # Heterogeneous Graph
    # 两种类型 oil and water well
    # 1-81: oil well; 82-132 water well
    graph_file = open(graph_path, 'r', encoding='utf-8')
    oil_num = 81
    water_num = 51
    A_00 = np.zeros((oil_num, oil_num))
    A_01 = np.zeros((oil_num, water_num))
    A_10 = np.zeros((water_num, oil_num))
    A_11 = np.zeros((water_num, water_num))

    for line in graph_file.readlines():
        x = int(line.split('\t')[0])
        y = int(line.split('\t')[1])
        # 加入距离特征
        dis = float(line.split('\t')[2])
        if x <= 81:
            if y <= 81:
                A_00[x - 1][y - 1] = dis
            else:
                A_01[x - 1][y - 82] = dis
        else:
            if y <= 81:
                A_10[x - 82][y - 1] = dis
            else:
                A_11[x - 82][y - 82] = dis

    A_00 = get_normalized_adj(A_00)
    A_01 = get_het_normalized_adj(A_01)
    A_10 = get_het_normalized_adj(A_10)
    A_11 = get_normalized_adj(A_11)

    
    return [A_00, A_01, A_10, A_11]


def load_dataset(graph_name, datadir, dataset):
    # Heterogeneous
    As = get_het_adj('data/graph.cites.' + graph_name)

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

    return adj_mat, X, As


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

def get_het_normalized_adj(A):
    Di = np.array(np.sum(A, axis=1)).reshape((-1,))
    Dj = np.array(np.sum(A, axis=0)).reshape((-1,))
    Di[Di <= 10e-5] = 10e-5
    Dj[Dj <= 10e-5] = 10e-5
    Dis = np.reciprocal(np.sqrt(Di))
    Djs = np.reciprocal(np.sqrt(Dj))
    A_wave = np.multiply(np.multiply(Dis.reshape((-1, 1)), A),
                        Djs.reshape((1, -1)))

    return A_wave
