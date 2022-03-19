import torch
import argparse
import torch.nn as nn
import time
import numpy as np
import os
import glob
import math

import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from utils import load_dataset, generate_dataset, get_normalized_adj
from model import HetSTGCN


num_timesteps_input = 20
num_timesteps_output = 1

patience = 100
batch_size = 1

parser = argparse.ArgumentParser(description="HetSTGCN")
parser.add_argument('--enable-cuda', action='store_true')
parser.add_argument('--datadir', type=str, default='./data/')
parser.add_argument('--dataset', type=str, default='STGCN163-1')
parser.add_argument('--graphname', type=str, default='threshold')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--data', type=int, default=132)
args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

train_path = args.graphname + "_train/"
if not os.path.exists(train_path):
    os.makedirs(train_path)


def train_epoch(train_input, train_target, batch_size, net):
    permutation = torch.randperm(train_input.shape[0])
    l_sum, n = 0.0, 0
    
    for i in range(0, train_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        X_batch, y_batch = train_input[indices], train_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
    
        l_sum += loss.item()
        n += 1

    return l_sum / n

def plot_data(predict, real, real_len, MSE, R2, gap, data_idx, path):
    predict = predict[::gap]
    real = real[::gap]
    x_index = [i for i in range(0, real_len - 1, gap)]

    plt.plot(x_index, predict, color='green', label='predict')
    plt.plot(x_index, real, color='red', label='real')
    plt.title('MSE:' + str(MSE) + '\n' + 'R2:' + str(R2))
    plt.legend()
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel("time")
    plt.ylabel("value")
    if predict.max() > real.max():
        scope_max = math.ceil(predict.max())
    else:
        scope_max = math.ceil(real.max())
    if scope_max <= 0:
        scope_max = 0
    if predict.min() > real.min():
        scope_min = math.floor(real.min())
    else:
        scope_min = math.floor(predict.min())
    if scope_min < 0:
        scope_min = 0
    pyplot.yticks(np.arange(scope_min, scope_max + 6, int((scope_max + 6 - scope_min) / 4)))
    pyplot.xticks(np.arange(0, real_len, 100))
    plt.savefig(path + str(args.graphname) + str(data_idx + 1) + '.jpg')
    plt.clf()


def start_test(all_input, all_target, test_input, test_target):
    print('Start Test: ')
    with torch.no_grad():
        net.eval()

        # 计算指标在测试集上
        test_input = test_input.to(device=args.device)
        test_target = test_target.to(device=args.device)

        test_out = net(test_input)
        test_loss = loss_criterion(test_out, test_target).to(device='cpu')
        print('Test Data Loss: {:.4f}'.format(test_loss.detach().numpy().item()))

        test_out = np.squeeze(test_out).cpu()
        test_target = np.squeeze(test_target).cpu()
        
        test_out_result = [[] for i in range(args.data)]
        test_target_result = [[] for i in range(args.data)]

        # 计算指标在所有数据上
        all_input = all_input.to(device=args.device)
        all_target = all_target.to(device=args.device)

        all_out = net(all_input)
        all_loss = loss_criterion(all_out, all_target).to(device='cpu')
        print('All Data Loss: {:.4f}'.format(all_loss.detach().numpy().item()))

        all_out = np.squeeze(all_out).cpu()
        all_target = np.squeeze(all_target).cpu()

        all_out_result = [[] for i in range(args.data)]
        all_target_result = [[] for i in range(args.data)]

        for i in range(args.data):
            test_out_result[i].append(np.squeeze(np.squeeze(test_out[:, i:i + 1])).detach().cpu().numpy())
            test_target_result[i].append(np.squeeze(np.squeeze(test_target[:, i:i + 1])).detach().cpu().numpy())

            all_out_result[i].append(np.squeeze(np.squeeze(all_out[:, i:i + 1])).detach().cpu().numpy())
            all_target_result[i].append(np.squeeze(np.squeeze(all_target[:, i:i + 1])).detach().cpu().numpy())

        TEST_MSE_SUM = []
        ALL_MSE_SUM = []
        for data_idx in range(args.data):
            test_predict = np.squeeze(test_out_result[data_idx])
            test_real = np.squeeze(test_target_result[data_idx])

            TEST_MSE = mean_squared_error(test_real, test_predict)
            TEST_MSE_SUM.append(TEST_MSE)
            TEST_RMSE = np.sqrt(TEST_MSE)
            TEST_MAE = mean_absolute_error(test_real, test_predict)
            TEST_R2 = r2_score(test_real, test_predict)

            all_predict = np.squeeze(all_out_result[data_idx])
            all_real = np.squeeze(all_target_result[data_idx])

            ALL_MSE = mean_squared_error(all_real, all_predict)
            ALL_MSE_SUM.append(ALL_MSE)
            ALL_RMSE = np.sqrt(ALL_MSE)
            ALL_MAE = mean_absolute_error(all_real, all_predict)
            ALL_R2 = r2_score(all_real, all_predict)

            test_real_len = len(test_predict) + 1
            all_real_len = len(all_predict) + 1

            # fo = open('{}/{}{}.txt'.format(args.graphname, args.graphname, data_idx + 1), 'w')
            # fo.write('results: ' + "Predict: " + str(data_predict) + '\n' +\
            #         "Real: " + str(data_real) + '\n' +\
            #         "MSE: " + str(MSE) + '\t' + \
            #         "RMSE: " + str(RMSE) + '\t' + \
            #         "MAE: " + str(MAE) + '\t' + \
            #         "R2: " + str(R2) + '\n')
            # fo.close()

            # fx = open('{}/{}MSE.txt'.format(args.graphname, args.graphname), "a+")
            # fx.write(str(MSE) + '\n')
            # fx.close()

            # fx = open('{}/{}RMSE.txt'.format(args.graphname, args.graphname), "a+")
            # fx.write(str(RMSE) + '\n')
            # fx.close()

            # fx = open('{}/{}MAE.txt'.format(args.graphname, args.graphname), "a+")
            # fx.write(str(MAE) + '\n')
            # fx.close()

            # fx = open('{}/{}R2.txt'.format(args.graphname, args.graphname), "a+")
            # fx.write(str(R2) + '\n')
            # fx.close()

            # 绘制测试数据的原始数值与预测数值
            plot_data(test_predict, test_real, test_real_len, TEST_MSE, TEST_R2, 1, data_idx, 'result/test/')

            # 绘制所有数据的原始数值与预测数值
            plot_data(all_predict, all_real, all_real_len, ALL_MSE, ALL_R2, 3, data_idx, 'result/all/')
        print('TEST MSE Mean: {}'.format(np.mean(TEST_MSE_SUM)))
        print('ALL MSE Mean: {}'.format(np.mean(ALL_MSE_SUM)))



if __name__ == '__main__':
    torch.manual_seed(7) # 具体是多少比较合适？

    A, X = load_dataset(args.graphname, args.datadir, args.dataset)

    all_input, all_target = generate_dataset(X, 
                                    num_timesteps_input, num_timesteps_output)

    split_line1 = int(X.shape[2] * 0.7)
    split_line2 = int(X.shape[2] * 0.8) # 7:1:2
    train_original_data = X[:, :, :split_line1]
    valid_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]
    train_input, train_target = generate_dataset(train_original_data,
                                    num_timesteps_input, num_timesteps_output)
    valid_input, valid_target = generate_dataset(valid_original_data,
                                    num_timesteps_input, num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                    num_timesteps_input, num_timesteps_output)

    A_wave = get_normalized_adj(A)
    A_wave = get_normalized_adj(A_wave)
    A_wave = torch.from_numpy(A_wave)

    net = HetSTGCN(A_wave.shape[0], A_wave, train_input.shape[3],
                    num_timesteps_input, num_timesteps_output).to(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_criterion = nn.MSELoss()


    best = args.epochs + 1
    best_epoch = 0
    bad_counter = 0

    t = time.time()
    for epoch in range(args.epochs):
        train_loss = train_epoch(train_input, train_target, batch_size, net)

        torch.save(net.state_dict(), '{}{}.pkl'.format(train_path, epoch))

        with torch.no_grad():
            net.eval()
            valid_input = valid_input.to(device=args.device)
            valid_target = valid_target.to(device=args.device)

            out = net(valid_input)
            valid_loss = loss_criterion(out, valid_target).to(device='cpu')
            valid_loss = valid_loss.detach().numpy().item()
            
            out_unnormalized = out.detach().cpu().numpy()
            target_unnormalized = valid_target.detach().cpu().numpy()
            valid_mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            
        print(  'Epoch: {:03d}'.format(epoch + 1), \
                'loss_train: {:.2f}'.format(train_loss),
                'loss_valid: {:.2f}'.format(valid_loss),
                'V_MAE: {:.4f}'.format(valid_mae),
                'time: {:.2f}s'.format(time.time() - t))

        if valid_loss < best:
            best = valid_loss
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t))

    print('Loading {}th epoch'.format(best_epoch))
    net.load_state_dict(torch.load('{}{}.pkl'.format(train_path, best_epoch)))
    
    start_test(all_input, all_target, test_input, test_target)



