# coding: utf-8

import os
import os.path
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import*
from copy import*
import sys
import pickle
import pandas as pd
import shutil
import torch.nn.functional as F
from Data import generate_N_targets_label

from Network import *

def drop_output(x, p):
    # TODO change it to be a class
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    if p == 0:
        p_distribut = torch.ones(x.size())
    else:
        binomial = torch.distributions.binomial.Binomial(probs=torch.tensor(1 - p))
        p_distribut = binomial.sample(x.size())
    return p_distribut

def define_unsupervised_target(output, N, device, Xth=None):
    with torch.no_grad():
        # define unsupervised target

        unsupervised_targets = torch.zeros(output.size(), device=device)

        # max_index = torch.zeros(output.size(), device=device)
        # other_index = torch.ones(output.size(), device=device)

        # N_maxindex
        if Xth != None:
            N_maxindex = torch.topk(output.detach() - Xth, N).indices
        else:
            N_maxindex = torch.topk(output.detach(), N).indices


        # # select the least responded neurons
        # k = int(0.5 * output.size(1))
        # lowval = output.topk(k, dim=1)[0][:, -1]
        # lowval = lowval.expand(output.shape[1], output.shape[0]).permute(1, 0)
        # comp = (output >= lowval).to(output)
        # unsupervised_targets = comp*output

        # max_index.scatter_(1, N_maxindex, torch.ones(output.size(), device=device))
        # other_index.scatter_(1, N_maxindex, torch.zeros(output.size(), device=device))
        # unsupervised_targets = torch.clamp(output.detach() * max_index * 10 + output.detach() * other_index * 0.1, 0, 1)

        unsupervised_targets.scatter_(1, N_maxindex, torch.ones(output.size(), device=device)) # WTA definition

        # print('the unsupervised vector is:', unsupervised_targets)

    return unsupervised_targets, N_maxindex

def smoothLabels(labels, smooth_factor, nudge_N):
    assert len(labels.shape) == 2, 'input should be a batch of one-hot-encoded data'
    assert 0 <= smooth_factor <= 1, 'smooth_factor should be between 0 and 1'

    if 0 <= smooth_factor <= 1:
        with torch.no_grad():
            # label smoothing
            labels *= 1 - smooth_factor
            labels += (nudge_N * smooth_factor) / labels.shape[1]
            # labels = drop_output(labels)
    else:
        raise ValueError('Invalid label smoothing factor: ' + str(smooth_factor))
    return labels


def classify(net, jparams, class_loader, k_select=None):
    # todo do the sum for each batch to save the memory
    net.eval()

    class_record = torch.zeros((jparams['n_class'], jparams['fcLayers'][-1]), device=net.device)
    labels_record = torch.zeros((jparams['n_class'], 1), device=net.device)
    for batch_idx, (data, targets) in enumerate(class_loader):

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        # forward propagation
        output = net(data.to(torch.float32))

        for i in range(jparams['n_class']):
            indice = (targets == i).nonzero(as_tuple=True)[0]
            labels_record[i] += len(indice)
            class_record[i, :] += torch.sum(output[indice, :], axis=0)

    # take the maximum activation as associated class
    class_moyenne = torch.div(class_record, labels_record)
    response = torch.argmax(class_moyenne, 0)
    # remove the unlearned neuron
    max0_indice = (torch.max(class_moyenne, 0).values == 0).nonzero(as_tuple=True)[0]
    response[max0_indice] = -1

    if k_select is None:
        return response
    else:
        k_select_neuron = torch.topk(class_moyenne, k_select, dim=1).indices.flatten()
        return response, k_select_neuron


def classify_network(net, class_net, layer_loader, optimizer, class_smooth):
    net.eval()

    class_net.train()

    # define the loss of classification layer
    criterion = torch.nn.CrossEntropyLoss()

    # create the list for training errors
    correct_train = torch.zeros(1, device=net.device).squeeze()
    total_train = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(layer_loader):
        optimizer.zero_grad()

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        # net forward
        with torch.no_grad():
            x = net(data.to(torch.float32)) # without the dropout

        # class_net forward
        output = class_net(x)
        # class label smooth
        if class_smooth:
            targets = smoothLabels(targets.to(torch.float32), 0.2, 1)
        loss = criterion(output, targets.to(torch.float32))
        # backpropagation
        loss.backward()
        optimizer.step()

        # calculate the training errors
        #prediction = torch.argmax(F.softmax(output, dim=1), dim=1)
        prediction = torch.argmax(output, dim=1)
        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def train_softmax(net, jparams, train_loader, epoch, optimizer):
    net.train()
    net.epoch = epoch + 1

    # construct the loss function
    if jparams['lossFunction'] == 'MSE':
        criterion = torch.nn.MSELoss()
    elif jparams['lossFunction'] == 'Cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()

    Xth = torch.zeros(jparams['fcLayers'][-1], device=net.device)

    # Stochastic mode
    if jparams['batchSize'] == 1:
        Y_p = torch.zeros(jparams['fcLayers'][-1], device=net.device)

    # pseudo_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if net.cuda:
            data = data.to(net.device)
            target = target.to(net.device)
            # if args.unlabeledPercent != 0 and args.unlabeledPercent != 1:

        optimizer.zero_grad()

        # forward propagation
        output = net(data.to(torch.float32))

        # use the softmax to define the unsupervised_targets
        unsupervised_targets = F.softmax(output.detach().clone(), dim=1)

        if jparams['Dropout']:
            target_activity = jparams['nudge_N'] / (jparams['fcLayers'][-1] * (
                    1 - jparams['dropProb'][-1]))  # dropout influences the target activity
        else:
            target_activity = jparams['nudge_N'] / jparams['fcLayers'][-1]

        if jparams['batchSize'] == 1:
            Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * unsupervised_targets[0]
            Xth += jparams['gamma'] * (Y_p - target_activity)
        else:
            Xth += jparams['gamma'] * (torch.mean(unsupervised_targets, axis=0) - target_activity)

        # calculate the loss on the gpu
        loss = criterion(output, unsupervised_targets.to(torch.float32))
        loss.backward()

        optimizer.step()

    return Xth

def train_Xth(net, jparams, train_loader, epoch, optimizer):

    net.train()
    net.epoch = epoch + 1

    # construct the loss function
    if jparams['lossFunction'] == 'MSE':
        criterion = torch.nn.MSELoss()
    elif jparams['lossFunction'] == 'Cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()
    # elif jparams['lossFunction'] == 'Ncross':
    #     criterion = Ncross_entropy_loss(jparams['nudge_N'])


    Xth = torch.zeros(jparams['fcLayers'][-1], device=net.device)

    # Stochastic mode
    if jparams['batchSize'] == 1:
        Y_p = torch.zeros(jparams['fcLayers'][-1], device=net.device)
    # unsupervised_correct = 0
    # pseudo_correct = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if net.cuda:
            data = data.to(net.device)
            # target = target.to(net.device)

        optimizer.zero_grad()

        # forward propagation
        output = net(data.to(torch.float32))
        # generate output mask
        output_mask = drop_output(output, p=jparams['dropProb'][-1]).to(net.device)
        output = output_mask*output

        # create the unsupervised target
        unsupervised_targets, N_maxindex = define_unsupervised_target(output, jparams['nudge_N'], net.device, Xth=Xth)
        # label smoothing
        if jparams['Smooth']:
            unsupervised_targets = smoothLabels(unsupervised_targets, 0.3, jparams['nudge_N'])
        unsupervised_targets = unsupervised_targets * output_mask
        target_activity = (1-jparams['dropProb'][-1])*jparams['nudge_N']/jparams['fcLayers'][-1]

        if jparams['batchSize'] == 1:
            Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * unsupervised_targets[0]
            Xth += jparams['gamma'] * (Y_p - target_activity)
        else:
            Xth += jparams['gamma'] * (torch.mean(unsupervised_targets, axis=0) - target_activity)

        # calculate the loss on the gpu
        loss = criterion(output, unsupervised_targets.to(torch.float32))
        loss.backward()

        optimizer.step()

    return Xth


def train_bp(net, jparams, train_loader, epoch, optimizer):

    net.train()
    net.epoch = epoch+1

    # construct the loss function
    if jparams['lossFunction'] == 'MSE':
        criterion = torch.nn.MSELoss()
    elif jparams['lossFunction'] == 'Cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()

    # create the list for training errors and testing errors
    correct_train = torch.zeros(1, device=net.device).squeeze()
    total_train = torch.zeros(1, device=net.device).squeeze()

    # for Homeostasis, initialize the moving average and the target activity
    # TODO consider the situation the batch number is not divisible

    for batch_idx, (data, targets) in enumerate(train_loader):

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)  # target here is the extern targets

        optimizer.zero_grad()

        # forward propagation
        output = net(data.to(torch.float32))
        # label smoothing
        if jparams['Smooth']:
            targets = smoothLabels(targets.to(torch.float32), 0.2, 1)

        # transform targets
        if jparams['fcLayers'][-1] > jparams['n_class']:
            number_per_class = jparams['fcLayers'][-1]//jparams['n_class']
            multi_targets = generate_N_targets_label(torch.argmax(targets, dim=1).tolist(), number_per_class, jparams['fcLayers'][-1])
            if net.cuda:
                multi_targets = multi_targets.to(net.device)
            loss = criterion(output, multi_targets.to(torch.float32))
        else:
            loss = criterion(output, targets.to(torch.float32))

        loss.backward()
        # print('the backward loss is:', net.W[0].weight.grad)
        optimizer.step()
        # count correct times for supervised BP

        # training error
        number_per_class = output.size(1) // 10
        prediction = torch.argmax(output, dim=1)//number_per_class

        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    # # update the lr after at the end of each epoch
    # scheduler.step()

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def test_Xth(net, jparams, test_loader, response, spike_record=0, output_record_path=None):
    '''
        Function to test the network
        '''
    net.eval()

    # record the total test time
    total_test = torch.zeros(1, device=net.device).squeeze()
    # records of errors for unsupervised BP
    correct_av_test = torch.zeros(1, device=net.device).squeeze()
    correct_max_test = torch.zeros(1, device=net.device).squeeze()

    # records of the spikes
    if spike_record:
        spike = torch.zeros(jparams['n_class'], jparams['fcLayers'][-1], device=net.device)
        predic_spike_max = torch.zeros(jparams['n_class'], jparams['n_class'], device=net.device)
        predic_spike_av = torch.zeros(jparams['n_class'], jparams['n_class'], device=net.device)

    # records of the output values for the test dataset
    if output_record_path is not None:
        df = pd.DataFrame()

    for batch_idx, (data, targets) in enumerate(test_loader):

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        if len(targets.size()) > 1: # for the training error
            targets = torch.argmax(targets, 1)

        output = net(data.to(torch.float32))
        if output_record_path is not None:
            d2 = {'img': output.cpu().tolist(), 'target': targets.cpu().tolist()}
            df2 = pd.DataFrame.from_records(d2)
            df = pd.concat([df, df2])

        # calculate the total testing times and record the testing labels
        total_test += targets.size()[0]

        # TODO make a version compatible with batch
        '''average value'''
        classvalue = torch.zeros(data.size(0), jparams['n_class'], device=net.device)

        for i in range(jparams['n_class']):
            indice = (response == i).nonzero(as_tuple=True)[0]
            if len(indice) == 0:
                classvalue[:, i] = -1
            else:
                classvalue[:, i] = torch.mean(output[:, indice], 1)

        predict_av = torch.argmax(classvalue, 1)
        correct_av_test += (predict_av == targets).sum().float()

        '''maximum value'''
        # remove the non response neurons
        non_response_indice = (response == -1).nonzero(as_tuple=True)[0]
        output[:, non_response_indice] = -1

        maxindex_output = torch.argmax(output, 1)
        predict_max = response[maxindex_output]
        correct_max_test += (predict_max == targets).sum().float()

        # spike record
        if spike_record:
            for batch_number in range(data.size(0)):
                spike[targets[batch_number], maxindex_output[batch_number]] += 1
                predic_spike_av[targets[batch_number], predict_av[batch_number]] += 1
                predic_spike_max[targets[batch_number], predict_max[batch_number]] += 1

    # save the output values:
    if output_record_path is not None:
        if os.name != 'posix':
            prefix = '\\'
        else:
            prefix = '/'
        df.to_pickle(str(output_record_path)+prefix+'output_records.pkl')
        del(df)

    # calculate the test error
    test_error_av = 1 - correct_av_test / total_test
    test_error_max = 1 - correct_max_test / total_test
    if spike_record:
        return test_error_av, test_error_max, spike, predic_spike_av, predic_spike_max
    return test_error_av, test_error_max


def test_bp(net, test_loader, output_record_path=None):
    '''
    Function to test the network
    '''
    net.eval()

    # record the total test time
    total_test = torch.zeros(1, device=net.device).squeeze()

    # records of accuracy for supervised BP
    correct_test = torch.zeros(1, device=net.device).squeeze()

    # records of the output values for the test dataset
    if output_record_path is not None:
        df = pd.DataFrame()

    for batch_idx, (data, targets) in enumerate(test_loader):

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        output = net(data.to(torch.float32))

        if output_record_path is not None:
            d2 = {'img': output.cpu().tolist(), 'target': targets.cpu().tolist()}
            df2 = pd.DataFrame.from_records(d2)
            df = pd.concat([df, df2])

        # calculate the total testing times and record the testing labels
        total_test += targets.size()[0]

        # calculate the accuracy
        # number_per_class = output.size(1) // 10
        # prediction = torch.argmax(output, dim=1) // number_per_class

        if output.size(1) == 10:
            prediction = torch.argmax(output, dim=1)
        else:
            # average prediction for multi-neurons per class
            class_average = torch.zeros((output.size(0), 10), device=net.device)
            neuron_per_class = output.size(1) // 10
            for i in range(10):
                class_average[:, i] = torch.mean(output[:, i:i + neuron_per_class], dim=1)
            prediction = torch.argmax(class_average, dim=1)

        correct_test += (prediction == targets).sum().float()

    # save the output values:
    if output_record_path is not None:
        if os.name != 'posix':
            prefix = '\\'
        else:
            prefix = '/'
        df.to_pickle(str(output_record_path) + prefix + 'output_records.pkl')
        del (df)

    test_error = 1 - correct_test / total_test
    return test_error


def test_unsupervised_layer(net, class_net, jparams, test_loader):
    net.eval()
    class_net.eval()

    # create the list for testing errors
    correct_test = torch.zeros(1, device=net.device).squeeze()
    total_test = torch.zeros(1, device=net.device).squeeze()
    loss_test = torch.zeros(1, device=net.device).squeeze()
    total_batch = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(test_loader):

        total_batch += 1
        targets = targets.type(torch.LongTensor)
        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        # record the total test
        total_test += targets.size()[0]

        # net forward
        x = net(data.to(torch.float32))
        # class_net forward
        output = class_net(x)
        # calculate the loss
        if jparams['class_activation'] == 'softmax' or jparams['class_activation'] == 'x':
            loss = F.cross_entropy(output, targets)
        else:
            loss = F.mse_loss(output, F.one_hot(targets, num_classes=jparams['n_class']))

        loss_test += loss.item()

        # calculate the training errors
        #prediction = torch.argmax(F.softmax(output, dim=1), dim=1)
        prediction = torch.argmax(output, dim=1)
        correct_test += (prediction == targets).sum().float()

    # calculate the test error
    test_error = 1 - correct_test / total_test
    loss_test = loss_test/total_batch

    return test_error, loss_test

def initDataframe(path, method='bp', dataframe_to_init='results.csv'):
    '''
    Initialize a dataframe with Pandas so that parameters are saved
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep=',', index_col=0)
    else:
        if method == 'bp':
            columns_header = ['Train_Error', 'Min_Train_Error', 'Test_Error', 'Min_Test_Error']
        elif method == 'bp_Xth':
            columns_header = ['One2one_av_Error', 'Min_One2one_av', 'One2one_max_Error', 'Min_One2one_max_Error']
        elif method == 'semi_supervised':
            columns_header = ['Unsupervised_Test_Error', 'Min_Unsupervised_Test_Error', 'Supervised_Test_Error', 'Min_Supervised_Test_Error']
        elif method == 'classification_layer':
            columns_header = ['Train_Class_Error', 'Min_Train_Class_Error', 'Final_Test_Error', 'Min_Final_Test_Error',
                              'Final_Test_Loss', 'Min_Final_Test_Loss']
        else:
            raise ValueError("The method {} is not defined".format(method))

        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(path + prefix + dataframe_to_init)
    return dataframe


def initXthframe(path, dataframe_to_init='Xth_norm.csv'):
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep=',', index_col=0)
    else:
        columns_header = ['Xth_norm']
        # if method == 'bp':
        #     columns_header = ['Train_Error', 'Min_Train_Error', 'Test_Error', 'Min_Test_Error']
        # else:
        #     columns_header = ['One2one_av_Error', 'Min_One2one_av', 'One2one_max_Error', 'Min_One2one_max_Error']

        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(path + prefix + 'Xth_norm.csv')
    return dataframe


def updateDataframe(BASE_PATH, dataframe, error1, error2, filename='results.csv', loss=None):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
    if loss is None:
        data = [error1[-1], min(error1), error2[-1], min(error2)]
    else:
        data = [error1[-1], min(error1), error2[-1], min(error2), loss[-1], min(loss)]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(BASE_PATH + prefix + filename)
    except PermissionError:
        input("Close the {} and press any key.".format(filename))

    return dataframe


def updateXthframe(BASE_PATH, dataframe, Xth_norm):

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    data = [Xth_norm[-1]]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(BASE_PATH + prefix + 'Xth_norm.csv')
    except PermissionError:
        input("Close the Xth_norm.csv and press any key.")

    return dataframe


def createPath():
    '''
    Create path to save data
    '''
    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    BASE_PATH +=  prefix + 'DATA-0'

    # BASE_PATH += prefix + args.dataset
    #
    # BASE_PATH += prefix + 'method-' + args.method
    #
    # BASE_PATH += prefix + args.action
    #
    # BASE_PATH += prefix + str(len(args.fcLayers)-2) + 'hidden'
    # BASE_PATH += prefix + 'hidNeu' + str(args.fcLayers[1])
    #
    # BASE_PATH += prefix + 'β-' + str(args.beta)
    # BASE_PATH += prefix + 'dt-' + str(args.dt)
    # BASE_PATH += prefix + 'T-' + str(args.T)
    # BASE_PATH += prefix + 'K-' + str(args.Kmax)
    #
    # BASE_PATH += prefix + 'Clamped-' + str(bool(args.clamped))[0]
    #
    # BASE_PATH += prefix + 'lrW-' + str(args.lrWeights)
    # BASE_PATH += prefix + 'lrB-' + str(args.lrBias)
    #
    # BASE_PATH += prefix + 'BaSize-' + str(args.batchSize)

    BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    print("len(BASE_PATH)="+str(len(BASE_PATH)))
    filePath = shutil.copy('plotFunction.py', BASE_PATH)

    files = os.listdir(BASE_PATH)

    if 'plotFunction.py' in files:
        files.pop(files.index('plotFunction.py'))

    if not files:
        BASE_PATH = BASE_PATH + prefix + 'S-1'
    else:
        tab = []
        for names in files:
            tab.append(int(names.split('-')[1]))
        BASE_PATH += prefix + 'S-' + str(max(tab)+1)


    os.mkdir(BASE_PATH)
    name = BASE_PATH.split(prefix)[-1]


    return BASE_PATH, name


def saveHyperparameters(args, BASE_PATH):
    '''
    Save all hyperparameters in the path provided
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
    f.write('Classic Equilibrium Propagation - Energy-based settings \n')
    f.write('   Parameters of the simulation \n ')
    f.write('\n')

    for key in args.__dict__:
        f.write(key)
        f.write(': ')
        f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()
