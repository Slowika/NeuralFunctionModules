from __future__ import print_function
import argparse
import os
import pickle
import random
import numpy as np
import csv
import tqdm

import torch
from torch.autograd import Variable
import time

from models.model import Preresnet_MLP, CNN_MLP_NFL

parser = argparse.ArgumentParser(description='NFL-based architectures')
parser.add_argument('--model', type=str, choices=['CNN_MLP_NFL', 'Preresnet_MLP'], default='CNN_MLP_NFL',
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 150)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--experiment_name', type=str, default='test',
                    help='Result folder name')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model=='Preresnet_MLP':
    model = Preresnet_MLP(args)
elif args.model=='CNN_MLP_NFL':
    model = CNN_MLP_NFL(args)


model_dirs = './{}'.format(args.experiment_name)
bs = args.batch_size

input_img = torch.FloatTensor(bs, 3, 75, 75) # 75 - the size
input_qst = torch.FloatTensor(bs, 11)
label = torch.LongTensor(bs)

if args.cuda:
    print("sending the inputs and targets to GPU")
    model = model.to(device)
    input_img = input_img.to(device)
    input_qst = input_qst.to(device)
    label = label.to(device)

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

def save_csv(name, folder, statistics, include_stat_names=False):
    """
    :param name: Csv file name, needs to include .csv
    :param folder:
    :param statistics:
    """
    csv_file_path = os.path.join(folder, name)

    #if include_stat_names:
    #    title_string = ",".join(list(str(statistics.keys())))

    #stats_string = ",".join(list(str(statistics.values())))

    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    with open(csv_file_path, 'a') as f:
        csv_writer = csv.DictWriter(f, statistics.keys())
        if include_stat_names:
            csv_writer.writeheader()
        else:
            csv_writer.writerow(statistics)

    return csv_file_path

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)

def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

def train(epoch, rel, norel):
    start_time = time.time()
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return

    random.shuffle(rel)
    random.shuffle(norel)

    stats = dict()

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    acc_norel = []
    acc_rel = []
    l_rel = []
    l_norel = []

    with tqdm.tqdm(total=len(rel[0]) // bs) as pbar_train:
        for batch_idx in range(len(rel[0]) // bs):
            tensor_data(rel, batch_idx)
            accuracy_rel, loss_rel = model.train_(input_img, input_qst, label)

            tensor_data(norel, batch_idx)
            accuracy_norel, loss_norel = model.train_(input_img, input_qst, label)

            acc_rel.append(accuracy_rel)
            acc_norel.append(accuracy_norel)
            l_rel.append(loss_rel.item())
            l_norel.append(loss_norel.item())

            # if batch_idx % args.log_interval == 0:
            #iter_string = 'Train Epoch: {} [{}/{} ({:.0f})] Relations accuracy: {:.0f} | Non-relations accuracy: {:.0f}' \
            #              ' | Relations loss:  {} | Non-relations loss: {}'.format(
            #                    epoch, batch_idx * bs * 2, len(rel[0]) * 2, 100. * batch_idx * bs / len(rel[0]), accuracy_rel,
            #                    accuracy_norel, loss_rel, loss_norel)
           # pbar_train.update(1)
           # pbar_train.set_description(iter_string)

        finish_time = time.time()
    stats['Epoch'] = epoch
    stats['Mean_train_acc_rel'] = np.mean(acc_rel)
    stats['Mean_train_acc_nonrel'] = np.mean(acc_norel)
    stats['Mean_train_loss_rel'] = np.mean(l_rel)
    stats['Mean_train_loss_nonrel'] = np.mean(l_norel)

    stats['Std_train_acc_rel'] = np.std(acc_rel)
    stats['Std_train_acc_nonrel'] = np.std(acc_norel)
    stats['Std_train_loss_rel'] = np.std(l_rel)
    stats['Std_train_loss_nonrel'] = np.std(l_norel)

    stats['Elapsed_time'] = finish_time - start_time

    # print('Epoch {}: Mean training accuracy for rel. questions {}'.format(epoch, np.mean(np.array(acc_rel))))
    # print('Epoch {}: Mean training accuracy for non-rel. questions {}'.format(epoch, np.mean(np.array(acc_norel))))
    # print('Epoch {}: Mean training loss for rel. questions {}'.format(epoch, np.mean(np.array(l_rel))))
    # print('Epoch {}: Mean training loss for non-rel. questions {}'.format(epoch, np.mean(np.array(l_norel))))

    # print('Epoch {}: Std training accuracy for rel. questions {}'.format(epoch, np.std(np.array(acc_rel))))
    # print('Epoch {}: Std training accuracy for non-rel. questions {}'.format(epoch, np.std(np.array(acc_norel))))
    # print('Epoch {}: Std training loss for rel. questions {}'.format(epoch, np.std(np.array(l_rel))))
    # print('Epoch {}: Std training loss for non-rel. questions {}'.format(epoch, np.std(np.array(l_norel))))

    # print('Train Epoch: {} took {} ms'.format(epoch, t-start))
    if epoch == 1:
        save_csv('training_statistics.csv', args.experiment_name, stats, True)
    save_csv('training_statistics.csv', args.experiment_name, stats)


def test(epoch, rel, norel):
    stats = dict()
    stats['Epoch'] = epoch

    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        accuracy_rels.append(model.test_(input_img, input_qst, label))

        tensor_data(norel, batch_idx)
        accuracy_norels.append(model.test_(input_img, input_qst, label))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    #print('\n Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
        #accuracy_rel, accuracy_norel))
    stats['Rel_test_accuracy'] = accuracy_rel.item()
    stats['Nonrel_test_accuracy'] = accuracy_norel.item()

    if epoch == 1:
        save_csv('test_statistics.csv', args.experiment_name, stats, True)
    save_csv('test_statistics.csv', args.experiment_name, stats)


def load_data():
    print('loading data...')
    dirs = './data/sortofclevr/'
    filename = os.path.join(dirs, 'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
        train_datasets, test_datasets = pickle.load(f, encoding='latin1')
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(relations[0], relations[1]):
            rel_train.append((img, qst, ans))
        for qst, ans in zip(norelations[0], norelations[1]):
            norel_train.append((img, qst, ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(relations[0], relations[1]):
            rel_test.append((img, qst, ans))
        for qst, ans in zip(norelations[0], norelations[1]):
            norel_test.append((img, qst, ans))

    return (rel_train, rel_test, norel_train, norel_test)


rel_train, rel_test, norel_train, norel_test = load_data()

print("Number of relational test examples: {}".format(len(rel_test)))
print("Number of non-relational test examples: {}".format(len(norel_test)))
print("Number of relational training examples: {}".format(len(rel_train)))
print("Number of non-relational training examples: {}".format(len(norel_train)))

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

for epoch in range(1, args.epochs + 1):
    train(epoch, rel_train, norel_train)
    test(epoch, rel_test, norel_test)
    model.save_model(epoch)
