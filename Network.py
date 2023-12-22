from scipy import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from activations import *
# define activation function

# TODO make the rho function as an NN class
class MLP(nn.Module):
    def __init__(self, jparams):
        super(MLP, self).__init__()

        self.batchSize = jparams['batchSize']
        self.eta = jparams['eta']
        self.output_num = jparams['fcLayers'][-1]
        self.fcLayers = jparams['fcLayers']

        self.W = nn.ModuleList(None)

        # put model on GPU is available and asked
        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device

        # # construct dropout layers
        # self.drop_layers = nn.ModuleList(None)
        # for i in range(len(jparams['dropProb'])):
        #     self.drop_layers.extend([nn.Dropout(p=float(jparams['dropProb'][i]))])

        # TODO redefine the convolutional layer with new hyper-parameters

        # construct fully connected networks
        self.fcnet = torch.nn.Sequential()

        for i in range(len(jparams['fcLayers'])-1):
            self.fcnet.add_module("rho_" + str(i), func_dict[jparams['activation_function'][i]]())
            self.fcnet.add_module("drop_"+str(i), nn.Dropout(p=float(jparams['dropProb'][i])))
            w = nn.Linear(jparams['fcLayers'][i], jparams['fcLayers'][i + 1], bias=True)
            nn.init.xavier_uniform_(w.weight, gain=0.5)
            nn.init.zeros_(w.bias)
            self.fcnet.add_module("fc_"+str(i), w)
        self.fcnet.add_module("rho_"+str(len(jparams['fcLayers'])-1), func_dict[jparams['activation_function'][-1]]())

        # self.drop_output = nn.Dropout(p=float(jparams['dropProb'][-1]))
        # self.fcnet.add_module("drop_"+str(len(jparams['fcLayers'])-1), self.drop_output)

        # # Prune at the initialization
        # TODO remove the Pruning??
        if jparams['Prune'] == 'Initialization':
            # for i in range(len(jparams['pruneAmount'])-1):
            if self.convNet:
                for i in range(self.conv_number):
                    prune.random_unstructured(self.Conv[i], name='weight', amount=jparams['pruneAmount'][i])
                    # prune.remove(self.convNet[i], name='weight')
                for i in range(len(jparams['fcLayers']) - 1):
                    prune.random_unstructured(self.W[i], name='weight', amount=jparams['pruneAmount'][i+self.conv_number])
            else:
                for i in range(len(jparams['fcLayers']) - 1):
                    prune.random_unstructured(self.W[i], name='weight', amount=jparams['pruneAmount'][i])

        self = self.to(device)

    def forward(self, x):
        return self.fcnet(x)

    # TODO to remove this function and write the new one of semi-supervised learning
    def alter_N_target_sm(self, output, target, supervised_response, N):
        '''
        limited only at output=10, sm
        '''

        # define unsupervised target
        alter_targets = torch.zeros(output.size(), device=self.device)

        if target[0] == -1:
            N_maxindex = torch.topk(output.detach(), N).indices
            alter_targets.scatter_(1, N_maxindex, torch.ones(output.size(), device=self.device))
        else:
            supervised_index = (supervised_response == target[0]).nonzero(as_tuple=True)[0]
            if len(supervised_index) > 0:
                alter_targets.scatter_(1, supervised_index.reshape(output.size(0), -1), torch.ones(output.size(), device=self.device))
            else:
                N_maxindex = torch.topk(output.detach(), N).indices
                supervised_response[N_maxindex.item()] = target[0]
                alter_targets.scatter_(1, N_maxindex, torch.ones(output.size(), device=self.device))

        return alter_targets, supervised_response


# TODO do the sequential network for the cnn
class CNN(nn.Module):
    def __init__(self, jparams):
        super(CNN, self).__init__()
        self.conv_number = 0
        if jparams['dataset'] == 'mnist':
            input_size = 28
        elif jparams['dataset'] == 'cifar10':
            input_size = 32
        else:
            raise ValueError("The convolutional network now is only designed for mnist dataset")

        # # add batch normalization layer
        # self.batch1d = [nn.BatchNorm1d(self.output_num, affine=True,
        #                                device=self.device, dtype=None)]

        self.Conv = nn.ModuleList(None)  # TODO tobe removed?

        conv_number = int(len(jparams['C_list']) - 1)
        self.conv_number = conv_number
        self.fcLayers = jparams['fcLayers']
        self.convNet = nn.Sequential()
        # construct the convolutional layer
        self.convNet.add_module("rho_0", func_dict[jparams['activation_function'][0]]())
        self.convNet.add_module("drop_0", nn.Dropout(p=float(jparams['dropProb'][0])))
        for i in range(conv_number):
            # if jparams['batchNorm'][i]:
            #     self.convNet.add_module("batchN_"+str(i), nn.BatchNorm2d(jparams['C_list'][i]))
            self.convNet.add_module("conv_" + str(i), nn.Conv2d(in_channels=jparams['C_list'][i], out_channels=jparams['C_list'][i + 1],
                                       kernel_size=jparams['convF'][i], stride=1,
                                       padding=jparams['Pad'][i]))
            self.convNet.add_module("rho_"+str(i+1), func_dict[jparams['activation_function'][i+1]]())
            if jparams['batchNorm'][i]:
                self.convNet.add_module("batchN_"+str(i), nn.BatchNorm2d(jparams['C_list'][i+1]))
            self.convNet.add_module("drop_"+str(i+1), nn.Dropout(p=float(jparams['dropProb'][i+1])))
            # define different pool
            if jparams['pool_type'][i] == 'max':
                self.convNet.add_module("Pool_"+str(i), nn.MaxPool2d(kernel_size=jparams['Fpool'][i], stride=jparams['Spool'][i]))
            elif jparams['pool_type'][i] == 'av':
                self.convNet.add_module("Pool_"+str(i), nn.AvgPool2d(kernel_size=jparams['Fpool'][i], stride=jparams['Spool'][i]))

        # self.convNet.add_module("rho_" + str(conv_number), func_dict[jparams['activation_function'][conv_number]]())
        # for i in range(conv_number):
        #     self.Conv.append(nn.Conv2d(in_channels=jparams['C_list'][i], out_channels=jparams['C_list'][i + 1],
        #                                kernel_size=jparams['convF'][i], stride=1,
        #                                padding=jparams['Pad'][i]))
        #
        # self.pool = nn.MaxPool2d(kernel_size=jparams['Fpool'], stride=jparams['Fpool'])
        # self.avPool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.ReLu = nn.ReLU()

        size_convpool_list = [input_size]

        for i in range(conv_number):
            size_convpool_list.append(int(np.floor(
                (size_convpool_list[i] - jparams['convF'][i] + 1 + 2 * jparams['Pad'][i] - jparams['Fpool'][i]) / jparams['Spool'][i] + 1)))  # the size after the pooling layer

        # size_convpool_list.append(int(np.floor(
        #     (size_convpool_list[-1] - jparams['convF'][-1] + 1 + 2 * jparams['Pad'][-1] - 2) / 2 + 1)))

        conv_output = jparams['C_list'][-1] * size_convpool_list[-1] ** 2

        # define the fully connected layer
        self.fcLayers.insert(0, conv_output)
        self.fcNet = nn.Sequential()

        for i in range(len(jparams['fcLayers']) - 1):
            # self.fcNet.add_module("drop_" + str(i+conv_number), nn.Dropout(p=float(jparams['dropProb'][i+conv_number])))
            self.fcNet.add_module("fc_" + str(i),
                                  nn.Linear(jparams['fcLayers'][i], jparams['fcLayers'][i + 1], bias=True))
            self.fcNet.add_module("rho_" + str(i+conv_number+1), func_dict[jparams['activation_function'][i+conv_number+1]]())
            if i != len(jparams['fcLayers'])-2:
                self.fcNet.add_module("drop_" + str(i + conv_number + 1), func_dict[jparams['dropProb'][i + conv_number+1]]())
        # last dropout layer generate outside the network
        # # define the Triangle layer
        # triangle = [Triangle(0.7), Triangle(1.4), Triangle(1)]
        # self.triangle = triangle
        #
        # # define the batchNormalize layer
        # batchNorm = []
        # for i in range(self.conv_number):
        #     batchNorm.append(nn.BatchNorm2d(jparams['C_list'][i]).to(self.device))
        # self.batchNorm = batchNorm
        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False
        self = self.to(device)
        self.device = device

    def forward(self, x):
        x = self.convNet(x)
        x = x.view(x.size(0), -1)
        x = self.fcNet(x)
        return x

        # if self.Dropout:
        #     x = self.drop_layers[0](x)
        #     # add the dropout
        #     if self.convNet:
        #         # # whitening
        #         # x = self.whiten(x)
        #         for i in range(self.conv_number):
        #             x = self.batchNorm[i](x)
        #             x = self.Conv[i](x)
        #             x = self.triangle[i](x)
        #             # x = F.relu(x)
        #             if i == self.conv_number-1 and self.Avpool > 0:
        #                 x = self.avPool(x)
        #             else:
        #                 x = self.pool(x)
        #             x = self.drop_layers[i+1](x)
        #
        #         # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        #         x = x.view(x.size(0), -1)
        #
        #     for i in range(len(self.fcLayers) - 1):
        #         x = self.rho(self.W[i](x), self.activation[i])
        #         x = self.drop_layers[i+self.conv_number+1](x)
        #
        # return x


    # TODO to remove this function and write the new one of semi-supervised learning
    def alter_N_target_sm(self, output, target, supervised_response, N):
        '''
        limited only at output=10, sm
        '''

        # define unsupervised target
        alter_targets = torch.zeros(output.size(), device=self.device)

        if target[0] == -1:
            N_maxindex = torch.topk(output.detach(), N).indices
            alter_targets.scatter_(1, N_maxindex, torch.ones(output.size(), device=self.device))
        else:
            supervised_index = (supervised_response == target[0]).nonzero(as_tuple=True)[0]
            if len(supervised_index) > 0:
                alter_targets.scatter_(1, supervised_index.reshape(output.size(0), -1), torch.ones(output.size(), device=self.device))
            else:
                N_maxindex = torch.topk(output.detach(), N).indices
                supervised_response[N_maxindex.item()] = target[0]
                alter_targets.scatter_(1, N_maxindex, torch.ones(output.size(), device=self.device))

        return alter_targets, supervised_response


class Classifier(nn.Module):
    # one layer perceptron does not need to be trained by EP
    def __init__(self, jparams):
        super(Classifier, self).__init__()
        # construct the classifier layer
        self.classifier = torch.nn.Sequential(nn.Dropout(p=float(jparams['class_dropProb'])),
                                              nn.Linear(jparams['fcLayers'][-1], jparams['n_class']),
                                              func_dict[jparams['class_activation']]())

        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)

    def forward(self, x):
        return self.classifier(x)


