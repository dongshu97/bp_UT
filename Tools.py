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

from Network import*


def classify(net, jparams, class_loader):

    net.eval()

    for batch_idx, (data, targets) in enumerate(class_loader):

        if net.cuda:
            data = data.to(net.device)
            # targets = targets.to(net.device)

        # forward propagation
        output = net(data.to(torch.float32))
        # make a copy in the device (it works even initially in cpu)
        # n_output = torch.clone(output).detach().cpu()

        # record all the output values
        if batch_idx == 0:
            result_output = output.detach()
        else:
            result_output = torch.cat((result_output, output.detach()), 0)

        # record all the class sent
        if batch_idx == 0:
            class_vector = targets
        else:
            class_vector = torch.cat((class_vector, targets), 0)

    ##################### classifier one2one ########################

    class_moyenne = torch.zeros((jparams['n_class'], jparams['fcLayers'][-1]), device=net.device)

    for i in range(jparams['n_class']):
        indice = (class_vector == i).nonzero(as_tuple=True)[0]
        result_single = result_output[indice, :]
        class_moyenne[i, :] = torch.mean(result_single, axis=0)

    # for the unclassified neurons, we kick them out from the responses
    unclassified = 0
    response = torch.argmax(class_moyenne, 0)
    max0_indice = (torch.max(class_moyenne, 0).values == 0).nonzero(as_tuple=True)[0]
    response[max0_indice] = -1
    unclassified += max0_indice.size(0)

    return response


def cluster(class_vector, result_output, n_class, device, k_select):

    responses = [[] for k in range(len(result_output))]
    max_response_neurons = [[] for k in range(len(result_output))]

    total_unclassifited = []

    for i in range(len(result_output)):

        class_moyenne = torch.zeros((n_class, result_output[i][0].size()[1]), device=device)

        if k_select > result_output[i][0].size()[1]:
            raise ValueError("The presented neurons number is larger than the number of neurons in layer {}".format(i))

        for n in range(n_class):
            indice = (class_vector == n).nonzero(as_tuple=True)[0]
            result_single = result_output[i][0][indice, :]
            class_moyenne[n, :] = torch.mean(result_single, axis=0)

        # for the unclassified neurons, we kick them out from the responses
        unclassified = 0
        response_layer = torch.argmax(class_moyenne, 0)
        k_select_layer = torch.topk(class_moyenne, k_select, dim=1).indices # several classes may have one same neuron as their strongest response
        # TODO verify max0_indice
        max0_indice = (torch.max(class_moyenne, 0).values == 0).nonzero(as_tuple=True)[0]
        response_layer[max0_indice] = -1
        unclassified += max0_indice.size(0)
        # add the responses to blank list
        responses[i].append(response_layer)
        max_response_neurons[i].append(k_select_layer.flatten())

        total_unclassifited.append(unclassified)

    return responses, max_response_neurons, total_unclassifited


def classify_layers(net, jparams, class_loader, k_select):
    '''To the give the class for each hidden layer'''

    net.eval()

    if jparams['convNet']:
        layers_number = int(len(jparams['convLayers'])/5) + len(jparams['fcLayers']) - 1
        conv_number = int(len(jparams['convLayers'])/5)
    else:
        layers_number = len(jparams['fcLayers']) - 1
        conv_number = 0

    result_outputs = [[] for k in range(layers_number)]

    for batch_idx, (data, targets) in enumerate(class_loader):

        if net.cuda:
            data = data.to(net.device)
            #targets = targets.to(net.device)

        # record all the class sent
        if batch_idx == 0:
            class_vector = targets
        else:
            class_vector = torch.cat((class_vector, targets), 0)

        # give the data before the cycle
        x = data

        # redo the forward propagation
        if net.convNet:
            for i in range(conv_number):
                x = net.conv_number[i](x)
                x = F.relu(x),
                x = net.pool(x)
                #x = x.view(x.size(0), -1)
                # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
                output = x.clone().view(x.size(0), -1)

                if batch_idx == 0:
                    result_outputs[i].append(output.detach())
                else:
                    result_outputs[i][0] = torch.cat((result_outputs[i][0], output.detach()), 0)

        for i in range(len(jparams['fcLayers'])-1):
            x = net.rho(net.W[i](x), jparams['activation_function'][i])
            output = x.clone()

            if batch_idx == 0:
                result_outputs[i + conv_number].append(output.detach())
            else:
                result_outputs[i + conv_number][0] = torch.cat((result_outputs[i + conv_number][0], output.detach()), 0)

    ##################### classifier one2one ########################
    all_responses, max_response_neurons, total_unclassified = cluster(class_vector, result_outputs, jparams['n_class'], net.device, k_select)

    return all_responses, max_response_neurons, total_unclassified


def classify_network(net, class_net, jparams, layer_loader):
    net.eval()
    class_net.train()

    # define the loss of classification layer
    if jparams['class_activation'] == 'softmax':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    parameters = list(class_net.parameters())

    # create the list for training errors
    correct_train = torch.zeros(1, device=net.device).squeeze()
    total_train = torch.zeros(1, device=net.device).squeeze()

    # construct the optimizer
    if jparams['class_Optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=jparams['class_lr'])
    elif jparams['class_Optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=jparams['class_lr'])

    for batch_idx, (data, targets) in enumerate(layer_loader):
        optimizer.zero_grad()

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        # net forward
        with torch.no_grad():
            x = net(data.to(torch.float32))
        # class_net forward
        output = class_net.forward(x)
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


def train_Xth(net, jparams, train_loader, epoch, optimizer, supervised_response=None):

    net.train()
    net.epoch = epoch + 1

    # construct the loss function
    if jparams['lossFunction'] == 'MSE':
        criterion = torch.nn.MSELoss()
    elif jparams['lossFunction'] == 'Cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()

    Xth = torch.zeros(jparams['fcLayers'][-1], device=net.device)
    # # construct the layer-wise parameters
    # layer_names = []
    # for idx, (name, param) in enumerate(net.named_parameters()):
    #     layer_names.append(name)
    #     # print(f'{idx}: {name}')
    #
    # parameters = []
    # # prev_group_name = layer_names[0].split('.')[0] + '.' + layer_names[0].split('.')[1]
    #
    # for idx, name in enumerate(layer_names):
    #
    #     # parameter group name
    #     # cur_group_name = name.split('.')[0] + '.' + name.split('.')[1]
    #
    #     # update learning rate
    #     if idx % 2 == 0:
    #         lr_indx = int(idx / 2)
    #         lr = jparams['lr'][lr_indx]
    #
    #     # display info
    #     # print(f'{idx}: lr = {lr:.6f}, {name}')
    #
    #     # append layer parameters
    #     parameters += [{'params': [p for n, p in net.named_parameters() if n == name and p.requires_grad],
    #                     'lr': lr}]
    #
    # # construct the optimizer
    # if jparams['Optimizer'] == 'SGD':
    #     optimizer = torch.optim.SGD(parameters)
    # elif jparams['Optimizer'] == 'Adam':
    #     optimizer = torch.optim.Adam(parameters)

    # Stochastic mode
    if jparams['batchSize'] == 1:
        Y_p = torch.zeros(jparams['fcLayers'][-1], device=net.device)
    # unsupervised_correct = 0
    # pseudo_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if net.cuda:
            data = data.to(net.device)
            target = target.to(net.device)
            # if args.unlabeledPercent != 0 and args.unlabeledPercent != 1:

        optimizer.zero_grad()

        # # weight normalization
        # if args.WN:
        #     # TODO try the WN only at the beginning OR at each epoch
        #     net.Weight_normal(args)

        # forward propagation
        output = net(data.to(torch.float32))

        # create the unsupervised target
        unsupervised_targets, N_maxindex = net.define_unsupervised_target(output, jparams['nudge_N'], net.device, Xth=Xth)
        # pseudo_labels, peudo_maxindex = net.define_unsupervised_target(output, jparams['nudge_N'], net.device, Xth=None)
        # unsupervised_correct += (target == torch.argmax(unsupervised_targets, dim=1)).sum().float()
        # pseudo_correct += (target == torch.argmax(pseudo_labels, dim=1)).sum().float()
        # label smoothing
        unsupervised_targets = net.smoothLabels(unsupervised_targets, 0.2, jparams['nudge_N'])

        if jparams['Dropout']:
            target_activity = jparams['nudge_N'] / (jparams['fcLayers'][-1] * (
                    1 - jparams['dropProb'][-1]))  # dropout influences the target activity
        else:
            target_activity = jparams['nudge_N'] / jparams['fcLayers'][-1]

        if jparams['batchSize'] == 1:
            Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * unsupervised_targets[0]
            Xth += net.gamma * (Y_p - target_activity)
        else:
            Xth += net.gamma * (torch.mean(unsupervised_targets, axis=0) - target_activity)


        # calculate the loss on the gpu
        loss = criterion(output, unsupervised_targets.to(torch.float32))
        loss.backward()
        # if jparams['randomHidden']:
        #  # we keep only the gradients of output layer
        #     for i in range(len(net.W)-1):
        #         net.W[i].weight.grad = torch.zeros(net.W[i].weight.size(), device=net.device)
        #         net.W[i].bias.grad = torch.zeros(net.W[i].bias.size(), device=net.device)
        # print('the backward loss is:', net.W[0].weight.grad)

        optimizer.step()


        # # update the lr after at the end of each epoch
        # scheduler.step()
    # if args.unlabeledPercent != 0 and args.unlabeledPercent != 1:
    #     return Xth, supervised_response
    # else:
    # print('unsupervised correct is:', unsupervised_correct)
    # print('pseudo correct is:', pseudo_correct)
    return Xth


def train_bp(net, jparams, train_loader, epoch, optimizer):

    net.train()
    net.epoch = epoch+1

    # construct the loss function
    if jparams['lossFunction'] == 'MSE':
        criterion = torch.nn.MSELoss()
    elif jparams['lossFunction'] == 'Cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()

    # # construct the layer-wise parameters
    # layer_names = []
    # for idx, (name, param) in enumerate(net.named_parameters()):
    #     layer_names.append(name)
    #     #print(f'{idx}: {name}')
    #
    # parameters = []
    # #prev_group_name = layer_names[0].split('.')[0] + '.' + layer_names[0].split('.')[1]
    #
    # for idx, name in enumerate(layer_names):
    #
    #     # parameter group name
    #     #cur_group_name = name.split('.')[0] + '.' + name.split('.')[1]
    #
    #     # update learning rate
    #     if idx % 2 == 0:
    #         lr_indx = int(idx/2)
    #         lr = jparams['lr'][lr_indx]
    #
    #     # display info
    #     #print(f'{idx}: lr = {lr:.6f}, {name}')
    #
    #     # append layer parameters
    #     parameters += [{'params': [p for n, p in net.named_parameters() if n == name and p.requires_grad],
    #                     'lr': lr}]
    #
    # # construct the optimizer
    # # TODO changer optimizer to ADAM
    # if jparams['Optimizer'] == 'SGD':
    #     optimizer = torch.optim.SGD(parameters, momentum=0.9)
    # elif jparams['Optimizer'] == 'Adam':
    #     optimizer = torch.optim.Adam(parameters)
    # #optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # construct the scheduler
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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

        # # weight normalization
        # if args.WN:
        #     # TODO try the WN only at the beginning OR at each epoch
        #     net.Weight_normal(args)

        # forward propagation
        output = net(data.to(torch.float32))
        # label smoothing
        targets = net.smoothLabels(targets.to(torch.float32), 0.2, 1)

        loss = criterion(output, targets.to(torch.float32))

        # # <editor-fold desc= "WTA branch">
        # elif method == 'bp_wta':
        #     unsupervised_targets, maxindex = net.defi_target_01(output, args.nudge_N)
        #     loss = criterion(output, unsupervised_targets.to(torch.float32))
        # # </editor-fold>

        loss.backward()
        # print('the backward loss is:', net.W[0].weight.grad)
        optimizer.step()
        # count correct times for supervised BP
        # training errors is calculated during the training process??
        prediction = torch.argmax(output, dim=1)
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
            #TODO need to consider the situation that one class is not presented
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

        prediction = torch.argmax(output, dim=1)
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

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        # record the total test
        total_test += targets.size()[0]

        # net forward
        x = net(data.to(torch.float32))
        # class_net forward
        output = class_net.forward(x)
        # calculate the loss
        if jparams['class_activation'] == 'softmax':
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
        elif method == 'semi-supervised':
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
