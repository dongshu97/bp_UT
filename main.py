import os
import argparse
import json
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision
import torch.nn as nn
import torch.optim as optim
import pickle
import datetime
import numpy as np
import random
import platform
import time
from tqdm import tqdm
import pathlib
from Network import *
from Tools import *
from visu import *
from Data import *

parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'.',
    # default=r'D:\Results_data\BP_perceptron_without_dropout\784-100',
    help='path of json configuration'
)
parser.add_argument(
    '--trained_path',
    type=str,
    default=r'.',
    # default=r'D:\Results_data\BP_perceptron_without_dropout\784-100\S-11',
    help='path of model_dict_state_file'
)
args = parser.parse_args()

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

with open(args.json_path + prefix + 'config.json') as f:
  jparams = json.load(f)


# give the reproducible result if 'torchSeed' is not zero
if jparams['torchSeed']:
    torch.manual_seed(jparams['torchSeed'])

# define the two batch sizes
# TODO delete the variable of batchSize??
batch_size = jparams['batchSize']
batch_size_test = jparams['test_batchSize']

# <editor-fold desc="Prepare MNIST dataset">
if jparams['dataset'] == 'mnist':

    print('We use the MNIST Dataset')
    # Define the Transform
    if jparams['convNet']:
        transforms = [torchvision.transforms.ToTensor()]
    else:
        transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

    # Download the MNIST dataset
    if jparams['action'] == 'bp_Xth' or jparams['action'] == 'test' or jparams['action'] == 'visu':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                           transform=torchvision.transforms.Compose(transforms))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)

    elif jparams['action'] == 'bp':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms),
                                               target_transform=ReshapeTransformTarget(10))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)

    # Load the dataset
    # TODO change the target (do not use the ReshapeTransformTarget)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=jparams['test_batchSize'], shuffle=True)

    # define the class dataset
    seed = 34

    # create the class
    x = train_set.data
    y = train_set.targets

    classLabel_percentage = jparams['classLabel_percentage']
    if jparams['classLabel_percentage'] == 1:
        class_set = train_set
        layer_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms),
                                               target_transform=ReshapeTransformTarget(10))
    else:
        class_set = splitClass(x, y, classLabel_percentage, seed=seed,
                               transform=torchvision.transforms.Compose(transforms))

        layer_set = splitClass(x, y, classLabel_percentage, seed=seed,
                               transform=torchvision.transforms.Compose(transforms),
                               target_transform=ReshapeTransformTarget(10))

    # class_set = ClassDataset(root='./MNIST_class_seed', test_set=test_set, seed=seed,
    #                          transform=torchvision.transforms.Compose(transforms))

    class_loader = torch.utils.data.DataLoader(class_set, batch_size=jparams['test_batchSize'], shuffle=True)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=jparams['test_batchSize'], shuffle=True)

    # for batch_idx, (data, targets) in enumerate(train_loader):
    #     data_average = torch.mean(torch.norm(data,dim=1))
    #     print('the average of dataset is:', data_average)

# </editor-fold>
# TODO CIFAR-10 to be re-modified
elif jparams['dataset'] == 'cifar10':

    print('We use the CIFAR10 dataset')

    # Define the Transform
    if jparams['convNet']:
        transform_type = transforms.ToTensor()
    else:
        transform_type = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])

    # Download the CIFAR10 dataset
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_type)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_type,
                                            target_transform=ReshapeTransformTarget(10))
    seed = 1
    # TODO we may not use the target_transform to simply the calculation in unsupervised learning
    test_set.targets = np.array(test_set.targets)

    class_set = ClassDataset(root='./CIFAR_class_seed', test_set=test_set, seed=seed,
                             transform=transform_type,
                             target_transform=ReshapeTransformTarget(10))

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # load the datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=jparams['test_batchSize'], shuffle=True)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=1000, shuffle=True)

elif jparams['dataset'] == 'YinYang':
    print('We use the YinYang dataset')
    # 'YinYang' dataset is not compatible with YinYang
    if jparams['convNet']:
        raise ValueError("YinYang dataset do not apply to convolutional dataset ")

    if jparams['action'] == 'bp':
        train_set = YinYangDataset(size=5000, seed=42, target_transform=ReshapeTransformTarget(3))
    elif jparams['action'] == 'bp_Xth' or jparams['action'] == 'test' or jparams['action'] == 'visu':
        train_set = YinYangDataset(size=5000, seed=42)

    # validation_set = YinYangDataset(size=1000, seed=41)  # used for the hyperparameter research

    test_set = YinYangDataset(size=1000, seed=40)

    # the seed of classification is same as that of train/test
    class_set = YinYangDataset(size=1000, seed=42, sub_class=True)
    layer_set = YinYangDataset(size=1000, seed=42, sub_class=True, target_transform=ReshapeTransformTarget(3))

    # separate the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)
    # validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=jparams['test_batchSize'], shuffle=False)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=100, shuffle=False)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=100, shuffle=False)

if __name__ == '__main__':

    # mean_digits = torch.mean(torch.norm(torch.from_numpy(digits.data/16), dim=1))
    # print('the average norm of digits is:', mean_digits)

    # create the network
    net = Net(jparams)

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # set tensor default

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    # Create the data dossier
    BASE_PATH, name = createPath()

    # save hyper-parameters as json file
    with open(BASE_PATH + prefix + "config.json", "w") as outfile:
        json.dump(jparams, outfile)

    # <editor-fold desc="Train with supervised BP">
    if jparams['action'] == 'bp':

        print("Training the network with supervised bp")
        #saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, method='bp')
        print(DATAFRAME)

        train_error = []
        test_error = []

        # construct the layer-wise parameters
        layer_names = []
        for idx, (name, param) in enumerate(net.named_parameters()):
            layer_names.append(name)
            # print(f'{idx}: {name}')

        parameters = []

        for idx, name in enumerate(layer_names):

            # update learning rate
            if idx % 2 == 0:
                lr_indx = int(idx / 2)
                lr = jparams['lr'][lr_indx]

            # append layer parameters
            parameters += [{'params': [p for n, p in net.named_parameters() if n == name and p.requires_grad],
                            'lr': lr}]

        # construct the optimizer
        # TODO changer optimizer to ADAM
        if jparams['Optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(parameters, momentum=0.9)
        elif jparams['Optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(parameters)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        for epoch in tqdm(range(jparams['epochs'])):

            train_error_epoch = train_bp(net, jparams, train_loader, epoch, optimizer)
            train_error.append(train_error_epoch.item())

            # testing process
            test_error_epoch = test_bp(net, test_loader)
            test_error.append(test_error_epoch.item())

            scheduler.step()
            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, train_error, test_error)
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')
    # </editor-fold>

    # <editor-fold desc="Train with unsupervised Methods">
    elif jparams['action'] == 'bp_Xth':

        print("Training the unsupervised bp network")
        #saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, method='bp_Xth')
        print(DATAFRAME)

        #TODO to change it for the other uses

        # # load the pre-trianed network
        # net.load_state_dict(torch.load(
        #     r'D:\bp_convnet\DATA-0\2023-03-15\S-12\model_state_dict.pt'))

        # dataframe for Xth
        Xth_dataframe = initXthframe(BASE_PATH, 'Xth_norm.csv')

        test_error_av = []
        test_error_max = []
        X_th = []

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # since = time.time()
        # construct the layer-wise parameters
        layer_names = []
        for idx, (name, param) in enumerate(net.named_parameters()):
            layer_names.append(name)
            # print(f'{idx}: {name}')

        parameters = []

        for idx, name in enumerate(layer_names):

            # parameter group name

            # update learning rate
            if idx % 2 == 0:
                lr_indx = int(idx / 2)
                lr = jparams['lr'][lr_indx]

            # display info
            # print(f'{idx}: lr = {lr:.6f}, {name}')

            # append layer parameters
            parameters += [{'params': [p for n, p in net.named_parameters() if n == name and p.requires_grad],
                            'lr': lr}]

        # construct the optimizer
        # TODO changer optimizer to ADAM
        if jparams['Optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(parameters)
        elif jparams['Optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(parameters)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.5)

        for epoch in tqdm(range(jparams['epochs'])):
            # train process
            Xth = train_Xth(net, jparams, train_loader, epoch, optimizer)

            # classifying process
            response = classify(net, jparams, class_loader)

            # testing process
            test_error_av_epoch, test_error_max_epoch = test_Xth(net, jparams, test_loader, response=response, spike_record=0)

            test_error_av.append(test_error_av_epoch.item())
            test_error_max.append(test_error_max_epoch.item())
            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, test_error_av, test_error_max)

            X_th.append(torch.norm(Xth).item())
            Xth_dataframe = updateXthframe(BASE_PATH, Xth_dataframe, X_th)

            scheduler.step()
            # at each epoch, we update the model parameters
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')

        # create the classification layer
        class_net = Classlayer(jparams)

        # create dataframe for classification layer
        class_dataframe = initDataframe(BASE_PATH, method='classification_layer',
                                        dataframe_to_init='classification_layer.csv')
        torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict_0.pt')
        class_train_error_list = []
        final_test_error_list = []
        final_loss_error_list = []

        # at the end of unsupervised learning we train the final classification layer
        for epoch in tqdm(range(jparams['class_epoch'])):
            # we train the classification layer
            class_train_error_epoch = classify_network(net, class_net, jparams, layer_loader)
            class_train_error_list.append(class_train_error_epoch.item())
            # we test the final test error
            final_test_error_epoch, final_loss_epoch = test_unsupervised_layer(net, class_net, jparams, test_loader)
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = updateDataframe(BASE_PATH, class_dataframe, class_train_error_list, final_test_error_list,
                                              filename='classification_layer.csv', loss=final_loss_error_list)

            # save the trained class_net
            torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict.pt')

        # time_elapsed = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(
        #     time_elapsed // 60, time_elapsed % 60))

    # </editor-fold>

    # <editor-fold desc="Test a training methods with little dataset">
    elif jparams['action'] == 'test':
        '''plot the evolution of homeostasis term with a little subset'''

        iterations = 50
        train_iterator = iter(train_loader)
        test_iterator = iter(test_loader)
        class_iterator = iter(class_loader)

        # we re-load the trained network
        net.load_state_dict(torch.load(
            r'C:\Users\CNRS-THALES\OneDrive\文档\Homeostasis_python\bp_convNet\analysis\784-500\BP\model_state_dict.pt'))
        net.train()

        # Define the test methods
        test_method = 'bp_Xth'

        criterion = torch.nn.MSELoss()
        if jparams['Optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=jparams['lr'][0])
        elif jparams['Optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=jparams['lr'][0])

        if jparams['batchSize'] == 1:
            Y_p = torch.zeros(jparams['fcLayers'][-1], device=net.device)

        Xth_record = torch.zeros((1, jparams['fcLayers'][-1]), device=net.device)
        Yp_record = torch.zeros((1, jparams['fcLayers'][-1]), device=net.device)
        Xth = torch.zeros(jparams['fcLayers'][-1], device=net.device)
        targets_record = []

        for i in range(iterations):

            image, target = next(train_iterator)

            targets_record.append(target)

            if net.cuda:
                image = image.to(net.device)

            optimizer.zero_grad()

            # forward propagation
            output = net(jparams, image.to(torch.float32))

            unsupervised_targets = torch.zeros(output.size(), device=net.device)
            unsupervised_targets.scatter_(1, torch.topk(output.detach() - Xth, jparams['nudge_N']).indices,
                                          torch.ones(output.size(), device=net.device))

            target_activity = jparams['nudge_N'] / jparams['fcLayers'][-1]

            if jparams['batchSize'] == 1:
                Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * unsupervised_targets[0]
                Xth += jparams['gamma'] * (Y_p - target_activity)
            else:
                Y_p = torch.mean(unsupervised_targets, axis=0)
                Xth += jparams['gamma'] * (Y_p - target_activity)

            # calculate the loss on the gpu
            loss = criterion(output, unsupervised_targets.to(torch.float32))
            loss.backward()
            # print('the backward loss is:', net.W[0].weight.grad)
            optimizer.step()

            if i == 0:
                Xth_record[0, :] = Xth
                Yp_record[0, :] = Y_p
            else:
                Xth_record = torch.cat((Xth_record, Xth.reshape(1, jparams['fcLayers'][-1])), dim=0)
                Yp_record = torch.cat((Yp_record, Y_p.reshape(1, jparams['fcLayers'][-1])), dim=0)

        # Create the dossier to save the evolution figures
        path_test = pathlib.Path(BASE_PATH + prefix + 'evolution')
        path_test.mkdir(parents=True, exist_ok=True)

        if jparams['batchSize'] == 1:
            xlabel = 'Image number'
        else:
            xlabel = 'Batch number'

        plot_output(Xth_record, jparams['fcLayers'][-1], xlabel, 'Xth values', path_test, prefix)
        plot_output(Yp_record, jparams['fcLayers'][-1], xlabel, 'Neuron activities', path_test, prefix)
        #np.savetxt(path_test / 'target_record.txt', np.asarray(targets_record))
    # </editor-fold>

    # <editor-fold desc="Analyze a trained network">
    elif jparams['action'] == 'visu':
        # we re-load the trained network
        net.load_state_dict(torch.load(args.trained_path + prefix +'model_state_dict.pt'))

        net.eval()

        # we make the classification
        response = classify(net, jparams, class_loader)

        np.savetxt(pathlib.Path(BASE_PATH)/'response.txt', response.cpu().numpy())

        test_error_av_epoch, test_error_max_epoch, \
        spike, predic_spike_max, predic_spike_av = test_Xth(net, jparams, test_loader,
                                                           response=response, spike_record=1)
        print('the one2one av is :', test_error_av_epoch)
        print('the one2one max is :', test_error_max_epoch)
        one2one_result = [test_error_av_epoch.cpu().numpy(), test_error_max_epoch.cpu().numpy()]
        print(one2one_result)
        np.savetxt(BASE_PATH + prefix + 'one2one.txt', one2one_result, delimiter=',')

        # we train the classification layer
        class_net = Classlayer(jparams)

        # create dataframe for classification layer
        class_dataframe = initDataframe(BASE_PATH, method='classification_layer',
                                        dataframe_to_init='classification_layer.csv')
        torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict_0.pt')
        class_train_error_list = []
        final_test_error_list = []
        final_loss_error_list = []

        # at the end of unsupervised learning we train the final classification layer
        for epoch in tqdm(range(jparams['class_epoch'])):
            # we train the classification layer
            class_train_error_epoch = classify_network(net, class_net, jparams, layer_loader)
            class_train_error_list.append(class_train_error_epoch.item())
            # we test the final test error
            final_test_error_epoch, final_loss_epoch = test_unsupervised_layer(net, class_net, jparams, test_loader)
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = updateDataframe(BASE_PATH, class_dataframe, class_train_error_list, final_test_error_list,
                                              filename='classification_layer.csv', loss=final_loss_error_list)
            # save the trained class_net
            torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict.pt')

    # </editor-fold>

    elif jparams['action'] =='visuEP':
        # TODO modify to be compatible with the git code

        # # we reverse the input layers if we want to load the trained EP model
        # W_reverse = nn.ModuleList(None)
        # for i in range(len(jparams['fcLayers']) - 1):
        #     W_reverse.append(net.W[len(jparams['fcLayers']) - 2 - i])
        # net.W = W_reverse

        # we re-load the trained network
        # TODO this works only for the non-jit EP version, do also for the jit EP
        # net.load_state_dict(
        #     torch.load(r'D:\Results_data\Visualization_EP_batchmode\perceptron-500\error0.0667\model_state_dict.pt'))

        with open(r'D:\Results_data\Visualization_EP_batchmode\784-1024-1024\S-4\model_entire.pt', 'rb') as f:
            jit_net = torch.jit.load(f)

        with torch.no_grad():
            for i in range(len(jparams['fcLayers']) - 1):
                net.W[i].weight.data = jit_net.W[-i-1].transpose(0, 1)
                net.W[i].bias.data = jit_net.bias[-i-1]

        net.eval()

        # we make the classification
        response = classify(net, jparams, class_loader)

        np.savetxt(pathlib.Path(BASE_PATH) / 'response.txt', response.cpu().numpy())

        test_error_av_epoch, test_error_max_epoch, \
        spike, predic_spike_max, predic_spike_av = test_Xth(net, jparams, test_loader,
                                                            response=response, spike_record=1)
        print('the one2one av is :', test_error_av_epoch)
        print('the one2one max is :', test_error_max_epoch)

    # <editor-fold desc="Print out spikes">
    if jparams['spike']:
        # we create the dossier to save the spike figures
        path_spike = pathlib.Path(BASE_PATH + prefix + 'spike')
        path_spike.mkdir(parents=True, exist_ok=True)

        final_error_av, final_error_max, \
        spike, predic_spike_av, predic_spike_max = test_Xth(net, jparams, test_loader,
                                                            response=response, spike_record=1)

        # realize the response for the hidden neuron

        print('the one2one av is :', final_error_av)
        print('the one2one max is :', final_error_max)
        # figure the response
        if net.cuda:
            spike = spike.cpu()
            predic_spike_av = predic_spike_av.cpu()
            predic_spike_max = predic_spike_max.cpu()

        plot_spike(spike, 'Neuron responses', path_spike, prefix)
        plot_spike(predic_spike_av, 'Averaged predicted result', path_spike, prefix)
        plot_spike(predic_spike_max, 'Max predicted result', path_spike, prefix)
    # </editor-fold>

    # <editor-fold desc="Plot the weights of synapses">
    if jparams['imWeights']:

        # create the imShow dossier
        path_imshow = pathlib.Path(BASE_PATH + prefix + 'imShow')
        path_imshow.mkdir(parents=True, exist_ok=True)
        # TODO select the neurons to present te weight
        # for several layers structure
        for i in range(len(jparams['fcLayers'])-1):
            figName = 'layer' + str(i) + ' weights'
            display = jparams['display'][2*i:2*i+2]
            imShape = jparams['imShape'][2*i:2*i+2]
            weights = net.W[i].weight.data
            if jparams['device'] >= 0:
                weights = weights.cpu()
            plot_imshow(weights, jparams['fcLayers'][i+1], display, imShape, figName, path_imshow, prefix)

        # calculate the overlap matrix
        if len(jparams['fcLayers']) > 2:
            overlap = net.W[0].weight.data
            for j in range(len(jparams['fcLayers'])-2):
                overlap = torch.mm(net.W[j+1].weight.data, overlap)
            if jparams['device'] >= 0:
                overlap = overlap.cpu()
            display = jparams['display'][-2:]
            imShape = jparams['imShape'][0:2]
            plot_imshow(overlap, jparams['fcLayers'][-1], display, imShape, 'overlap', path_imshow, prefix)
    # </editor-fold>

    # <editor-fold desc="Visualize the network with Maximum activation image">
    if jparams['maximum_activation']:
        # TODO do parallelization for the image and select the neuron number at the beginning
        # return the responses of each layer

        k_select_neurons = jparams['select_num']

        all_responses, max_response_neurons, total_unclassified = classify_layers(net, jparams, class_loader, k_select_neurons)

        # create the maximum activation dossier
        path_activation = pathlib.Path(BASE_PATH + prefix + 'maxActivation')
        path_activation.mkdir(parents=True, exist_ok=True)

        if jparams['dataset'] == 'digits':
            data_average = 3.8638
            lr = 0.1
            nb_epoch = 100
        elif jparams['dataset'] == 'mnist':
            data_average = 9.20
            lr = 0.2
            nb_epoch = 500

        # This part does not apply the batch
        for j in range(len(jparams['fcLayers'])-1):
            if jparams['dataset'] == 'mnist':
                neurons_range = max_response_neurons[j][0].cpu().tolist()
                image_max = torch.zeros(len(neurons_range), 28 * 28)
                #image_max = torch.zeros(jparams['fcLayers'][j+1], 28*28)
                optimized_image = 0
                # image dimension (batch, channel, height, width)
                # image_max = torch.rand(args.fcLayers[j+1], 1, 28, 28, requires_grad=True, device=net.device)

            # for epoch in range(nb_epoch):
            #     optimizer = torch.optim.SGD([image_max], lr=lr)
            #     optimizer.zero_grad()
            #     output = image_max
            #     # TODO max activation image for the conv layers
            #     if args.convNet:
            #         output = net.conv1(output)
            #         output = net.conv2(output)
            #
            #     output = output.view(output.size(0), -1)
            #     for k in range(j + 1):
            #         activation = args.activation_function[k]
            #         output = net.rho(net.W[k](output), activation)
            #     # output size will be (batch, neuron_number), so it should be a identity matrix
            #     loss = 1-output
            #     # error !!!  grad can be implicitly created only for scalar outputs
                  # but what we really care is the diagonale one
            #     # we should update the grad for each neuron one by one
            #     loss.backward()
            #     optimizer.step()
            #
            #     image_max = (data_average * args.fcLayers[j+1] * (image_max.data / torch.norm(image_max.data).item())).requires_grad_(True)
            #
            # image_max = image_max.view(image_max.size(0), -1).cpu()

            # TODO the following part is the original model

            for i in neurons_range:
                #image = torch.rand(args.fcLayers[0], 1, requires_grad=True, device=net.device)
                if jparams['dataset'] == 'mnist':
                    image = torch.rand(1, 1, 28, 28, requires_grad=True, device=net.device)
                for epoch in range(nb_epoch):
                    optimizer = torch.optim.SGD([image], lr=lr)
                    optimizer.zero_grad()
                    output = image
                    # TODO max activation image for the conv layers
                    if jparams['convNet']:
                        output = net.conv1(output)
                        output = net.conv2(output)

                    output = output.view(output.size(0), -1)

                    for k in range(j+1):
                        if k == j:
                            activation = 'x'
                        else:
                            activation = jparams['activation_function'][k]

                        output = net.rho(net.W[k](output), activation)

                    # print(output)
                    # TODO try to update the loss by 1-output[0,i]

                    loss = -output[0, i]
                    loss.backward()
                    optimizer.step()
                    image = (data_average * (image.data / torch.norm(image.data).item())).requires_grad_(True)
                image_max[optimized_image, :] = torch.flatten(image).detach().cpu()
                optimized_image += 1
            figName = 'layer' + str(j) + ' max activation figures'
            imShape = jparams['imShape'][0:2]
            display = jparams['display'][2 * j:2 * j + 2]

            # calculate
            neurons_each_class = int(jparams['fcLayers'][j+1]/jparams['n_class'])

            plot_imshow(image_max, len(neurons_range), display, imShape, figName, path_activation, prefix)

            # plot_imshow(image_max, jparams['n_class'], display, imShape, figName, path_activation, prefix,
            #             all_responses[j][0])
            #
            # if neurons_each_class <= 5:
            #     plot_imshow(image_max, jparams['fcLayers'][j+1], display, imShape, figName, path_activation, prefix, all_responses[j][0])
            #
            # elif neurons_each_class <= 10:
            #     plot_NeachClass(image_max, jparams['n_class'], neurons_each_class, display, imShape,
            #                     all_responses[j][0], figName, path_activation, prefix)
            # else:
            #     # TODO change back the 2 to 10
            #     #plot_NeachClass(image_max, args.n_class, 10, imShape,all_responses[j][0], figName, path_activation, prefix)
            #     plot_NeachClass(image_max, jparams['n_class'], 2, display, imShape, all_responses[j][0], figName, path_activation,
            #                     prefix)

                # whether print the images for each single class

                # for q in range(args.n_class):
                #     figName_class = figName + ' for class ' + str(q)
                #     indices = (all_responses[j][0].cpu() == q).nonzero(as_tuple=True)[0].numpy()
                #     display_class = [4, 5]
                #
                #     plot_oneClass(image_max, display_class, imShape, indices, figName_class, path_activation, prefix)

    # </editor-fold>
