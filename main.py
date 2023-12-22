import os
import argparse
import json
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torch.optim as optim
import pickle
import datetime
import random
import platform
import time
from tqdm import tqdm
import pathlib
from Network import *
from Tools import *
from actions import *
from visu import *
from Data import *

parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'.',
    help='path of json configuration'
)
parser.add_argument(
    '--trained_path',
    type=str,
    # default=r'.',
    default=r'D:\bp_convnet\unsupervised_config\1layer\0.3_0.2_5e4_200decay\S-3',
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
    (train_loader, test_loader,
     class_loader, layer_loader,
     supervised_loader, unsupervised_loader) = returnMNIST(jparams, validation=False)

# TODO CIFAR-10 to be re-modified
elif jparams['dataset'] == 'cifar10':
    print('We use the CIFAR10 dataset')
    (train_loader, test_loader,
     class_loader, layer_loader,
     supervised_loader, unsupervised_loader) = returnCifar10(jparams, validation=False)

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
    if jparams['convNet']:
        net = CNN(jparams)
    else:
        net = MLP(jparams)

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # set tensor default

    # save hyper-parameters as json file
    BASE_PATH, name = createPath()

    with open(BASE_PATH + prefix + "config.json", "w") as outfile:
        json.dump(jparams, outfile)

    if jparams['action'] == 'bp':
        print("Training the network with supervised bp")
        supervised_bp(net, jparams, train_loader, test_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'bp_Xth':
        print("Training the unsupervised bp network")
        unsupervised_bp(net, jparams, train_loader, class_loader, test_loader, layer_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'semi_supervised':
        print("Training the model with semi_supervised learning")
        # trained_path = args.trained_path + prefix + 'model_pre_supervised_state_dict.pt'
        #
        trained_path = None
        semi_supervised_bp(net, jparams, supervised_loader, unsupervised_loader, test_loader, trained_path=trained_path, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'pre_supervised_bp':
        print("Training the model with pre_supervised learning")
        pre_supervised_bp(net, jparams, supervised_loader, test_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'train_class_layer':
        print("Train the supplementary class layer for unsupervised learning")
        trained_path = args.trained_path + prefix + 'model_state_dict.pt'
        # trained_path = None
        train_class_layer(net, jparams, layer_loader, test_loader, trained_path=trained_path, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'visuSemi':
        # we load the supervised pretraining network
        net.load_state_dict(torch.load(args.trained_path + prefix + 'model_pre_supervised_state_dict.pt', map_location=net.device))
        net.eval()
        # we register the output of supervised pretraining
        supervised_output_path = pathlib.Path(BASE_PATH + prefix + 'supervised')
        supervised_output_path.mkdir(parents=True, exist_ok=True)
        pretrain_error = test_bp(net, test_loader, output_record_path=supervised_output_path)
        print('pretrained supervised error is:', pretrain_error)

        # we load the final tranied network
        net.load_state_dict(torch.load(args.trained_path + prefix + 'model_semi_state_dict.pt', map_location=net.device))
        net.eval()
        # we register the output of final network
        semi_output_path = pathlib.Path(BASE_PATH + prefix + 'semiOut')
        semi_output_path.mkdir(parents=True, exist_ok=True)
        semi_error = test_bp(net, test_loader, output_record_path=semi_output_path)
        print('semi supervised error is:', semi_error)


    # <editor-fold desc="Analyze a trained network">
    elif jparams['action'] == 'visu':

        # we re-load the trained network
        net.load_state_dict(torch.load(args.trained_path + prefix +'model_state_dict.pt', map_location=net.device))
        net.eval()

        # create results file
        DATAFRAME = initDataframe(BASE_PATH, method='bp_Xth')
        print(DATAFRAME)
        test_error_av = []
        test_error_max = []

        # we make the classification
        response = classify(net, jparams, class_loader)

        np.savetxt(pathlib.Path(BASE_PATH)/'response.txt', response.cpu().numpy())

        # TODO try also to register the output values
        test_error_av_epoch, test_error_max_epoch = test_Xth(net, jparams, test_loader,
                                                           response=response, spike_record=0, output_record_path=BASE_PATH)
        # # to have the one2one training results
        # train_error_av_epoch, train_error_max_epoch = test_Xth(net, jparams, train_loader, response=response,
        #                                                        spike_record=0, output_record_path=BASE_PATH)

        print('the one2one av is :', test_error_av_epoch)
        print('the one2one max is :', test_error_max_epoch)

        one2one_result = [test_error_av_epoch.cpu().numpy(), test_error_max_epoch.cpu().numpy()]
        print(one2one_result)
        np.savetxt(BASE_PATH + prefix + 'one2one.txt', one2one_result, delimiter=',')

        test_error_av.append(test_error_av_epoch.item())
        test_error_max.append(test_error_max_epoch.item())
        DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, test_error_av, test_error_max)

        # we train the classification layer
        train_class_layer(net, jparams, layer_loader, test_loader, BASE_PATH=BASE_PATH)

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
        #TODO to show the weight of hidden layers
        # create the imShow dossier
        path_imshow = pathlib.Path(BASE_PATH + prefix + 'imShow')
        path_imshow.mkdir(parents=True, exist_ok=True)
        if jparams['convNet']:
            # show the receptive field
            figName = 'First convolutional layer'
            imShape = [jparams['convF'], jparams['convF']]
            for name, layer in net.convNet.named_children():
                if name == 'conv_0':
                    weights = layer.weight.data
            if jparams['device'] >= 0:
                weights = weights.cpu()
            plot_imshow(weights, jparams['fcLayers'][1], [4, 8], imShape, figName, path_imshow, prefix)

        else:
        # TODO select the neurons to present te weight
            # show the hidden layer
            figName = 'Hidden layer weights'
            display = jparams['display'][0:2]
            imShape = jparams['imShape'][0:2]
            for name, layer in net.fcnet.named_children():
                if name == 'fc_0':
                    weights = layer.weight.data
            if jparams['device'] >= 0:
                weights = weights.cpu()
            plot_imshow(weights, jparams['fcLayers'][1], display, imShape, figName, path_imshow, prefix)

        # for i in range(len(jparams['fcLayers'])-1):
        #     figName = 'layer' + str(i) + ' weights'
        #     display = jparams['display'][2*i:2*i+2]
        #     imShape = jparams['imShape'][2*i:2*i+2]
        #     for name, layer in net.fcnet.named_children():
        #         if name == 'fc_'+str(i):
        #             weights = layer.weight.data
        #     if jparams['device'] >= 0:
        #         weights = weights.cpu()
        #     plot_imshow(weights, jparams['fcLayers'][i+1], display, imShape, figName, path_imshow, prefix)

        # # calculate the overlap matrix # used for receptive field
        # if len(jparams['fcLayers']) > 2:
        #     overlap = net.fcnet['fc_0'].weight.data
        #     for j in range(len(jparams['fcLayers'])-2):
        #         overlap = torch.mm(net.fcnet['fc_'+str(i+1)].weight.data, overlap)
        #     if jparams['device'] >= 0:
        #         overlap = overlap.cpu()
        #     display = jparams['display'][-2:]
        #     imShape = jparams['imShape'][0:2]
        #     plot_imshow(overlap, jparams['fcLayers'][-1], display, imShape, 'overlap', path_imshow, prefix)
    # </editor-fold>

    if jparams['maximum_class']:
        # get the labels for output neurons
        response = classify(net, jparams, class_loader)
        # create the images path
        path_class = pathlib.Path(BASE_PATH + prefix + 'maxClass')
        path_class.mkdir(parents=True, exist_ok=True)
        if jparams['dataset'] == 'digits':
            data_average = 3.8638
            lr = 0.1
            nb_epoch = 100
        elif jparams['dataset'] == 'mnist':
            data_average = 9.20
            lr = 0.005
            nb_epoch = 100

        image_max = torch.zeros(jparams['n_class'], 28 * 28)

        indx_neurons = []
        indx_all_class = []
        max_neurons_per_class = jparams['fcLayers'][-1]  # change to the number of winners 'k'
        # neuron_per_class = int(display[0] * display[1] / jparams['n_class'])

        # select the neuron to be presented at the beginning
        for i in range(jparams['n_class']):
            index_i = (response.cpu() == i).nonzero(as_tuple=True)[0].numpy()
            np.random.shuffle(index_i)
            indx_all_class.append(index_i)
            max_neurons_per_class = min(max_neurons_per_class, len(index_i))

            # range_index = min(len(index_i), neuron_per_class)
            # indx_neurons.extend(index_i[0:range_index])

        indx_all_class_torch = torch.zeros([jparams['n_class'], max_neurons_per_class], dtype=torch.int64, device=net.device)

        # for i in range(jparams['n_class']):
        #     indx_all_class_torch[i, :] = torch.tensor(indx_all_class[i][0:max_neurons_per_class])

        for i in range(jparams['n_class']):
            indx_all_class_torch[i, :] = torch.tensor(indx_all_class[i][0:max_neurons_per_class])

        nudge_class_target = torch.zeros((jparams['n_class'], jparams['fcLayers'][-1]), requires_grad=False, device=net.device)
        nudge_class_target.scatter_(1, indx_all_class_torch,
                                        torch.ones((10, jparams['fcLayers'][-1]), requires_grad=False,
                                                   device=net.device))
        for i in range(jparams['n_class']):
            if jparams['dataset'] == 'mnist':
                image = torch.rand(1, 1, 28, 28, requires_grad=True, device=net.device)
            net.eval()
            for epoch in range(nb_epoch):
                optimizer = torch.optim.Adam([image], lr=lr)
                optimizer.zero_grad()
                output = image
                # # TODO max activation image for the conv layers
                if jparams['convNet']:
                    output = net.conv1(output)
                    output = net.conv2(output)

                output = output.view(output.size(0), -1)

                for k in range(len(jparams['fcLayers'])-1):
                    if k == len(jparams['fcLayers']) - 2:
                        activation = 'x'
                    else:
                        activation = jparams['activation_function'][k]
                    output = net.rho(net.W[k](output), activation)
                # if not jparams['convNet']:
                #     image = image.view(image.size(0), -1)
                #
                # output = output.view(output.size(0), -1)
                #output = net.forward(image)
                # for k in range(len(jparams['fcLayers'])-1):
                #     activation = jparams['activation_function'][k]
                #     output = net.rho(net.W[k](output), activation)

                # print(output)
                # TODO try to update the loss by 1-output[0,i]
                criterion = torch.nn.CrossEntropyLoss()
                nudge_class_target_perclass = nudge_class_target[i].reshape(1, -1).to(torch.float32)
                #loss = -output[0][i, :]
                loss = criterion(output, nudge_class_target_perclass)
                loss.backward()
                optimizer.step()
                image = (data_average * (image.data / torch.norm(image.data).item())).requires_grad_(True)

            image_max[i, :] = torch.flatten(image).detach().cpu()

        figName = 'max activation images for each class'
        imShape = jparams['imShape'][0:2]
        display = jparams['display'][-2:]

        # calculate
        plot_imshow(image_max, jparams['n_class'], display, imShape, figName, path_class, prefix)

    # <editor-fold desc="Visualize the network with Maximum activation image">
    if jparams['maximum_activation']:
        # TODO do parallelization for the image and select the neuron number at the beginning
        # return the responses of each layer
        responses, kmax_response_neuron = classify(net, jparams, class_loader, jparams['select_num'])

        # create the maximum activation dossier
        path_activation = pathlib.Path(BASE_PATH + prefix + 'maxActivation')
        path_activation.mkdir(parents=True, exist_ok=True)

        if jparams['dataset'] == 'digits':
            data_average = 3.8638
            lr = 0.1
            nb_epoch = 100
        elif jparams['dataset'] == 'mnist':
            data_average = 9.20
            lr = 12
            nb_epoch = 500

        # maximum activation function for the output layer
        if jparams['dataset'] == 'mnist':
            neurons_range = kmax_response_neuron.cpu().tolist()
            image_max = torch.zeros(len(neurons_range), 28 * 28)
            optimized_num = 0
            for neu in neurons_range:
                image = torch.rand(1, 28 * 28, requires_grad=True, device=net.device)
                optimizer = torch.optim.SGD([image], lr=lr)
                for epoch in range(nb_epoch):
                    optimizer.zero_grad()
                    net.eval()
                    # should not apply the last activation function
                    output = image
                    for i in range(len(net.fcnet)-2):
                        output = net.fcnet[i](output)
                    loss = - output[0, neu]
                    loss.backward()
                    optimizer.step()
                    image = (data_average * (image.data / torch.norm(image.data).item())).requires_grad_(True)

                image_max[optimized_num, :] = image.detach().cpu().clone()
                optimized_num += 1

        figName = 'last layer' + ' max activation figures'
        imShape = jparams['imShape'][0:2]
        display = jparams['display'][-2:]

        # calculate
        neurons_each_class = int(jparams['fcLayers'][-1]/jparams['n_class'])

        plot_imshow(image_max, len(neurons_range), display, imShape, figName, path_activation, prefix)

# </editor-fold>
