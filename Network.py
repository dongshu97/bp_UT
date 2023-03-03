from scipy import*
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, jparams):
        super(Net, self).__init__()

        self.batchSize = jparams['batchSize']
        self.eta = jparams['eta']
        self.output = jparams['fcLayers'][-1]
        self.convNet = jparams['convNet']
        self.fcLayers = jparams['fcLayers']
        self.rho = jparams['rho']

        # TODO redefine the convolutional layer with new hyper-parameters
        if self.convNet:
            self.Conv = nn.ModuleList(None)
            conv_number = int(len(jparams['convLayers']) / 5)
            self.conv_number = conv_number
            for i in range(conv_number):
                self.Conv.append(nn.Conv2d(in_channels=jparams['convLayers'][i*5], out_channels=jparams['convLayers'][i*5+1],
                                           kernel_size=jparams['convLayer'][i*5+2], stride=jparams['convLayer'][i*5+3],
                                           padding=jparams['convLayer'][i*5+4]))
            self.pool = nn.MaxPool2d(kernel_size=2)

            #
            # self.conv1 = nn.Sequential(
            #     nn.Conv2d(
            #         in_channels=1,
            #         out_channels=16,
            #         #out_channels=32,
            #         kernel_size=5,
            #         stride=1,
            #         padding=2,
            #         #padding=1,
            #     ),
            #     nn.ReLU(),
            #     nn.MaxPool2d(kernel_size=2),
            # )
            # self.conv2 = nn.Sequential(
            #     nn.Conv2d(16, 32, 5, 1, 2),
            #     #nn.Conv2d(32, 64, 5, 1, 1),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2),
            # )
        # fully connected layer, output 10 classes
        self.W = nn.ModuleList(None)
        with torch.no_grad():
            for i in range(len(jparams['fcLayers']) - 1):
                self.W.extend([nn.Linear(jparams['fcLayers'][i], jparams['fcLayers'][i + 1], bias=True)])

        # put model on GPU is available and asked
        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)

    def rho(self, x, type):
        if type == 'relu':
            return F.relu(x)
        elif type == 'softmax':
            return F.softmax(x, dim=1)
        elif type == 'x':
            return x
        elif type == 'sigmoid':
            return torch.sigmoid(x)
        elif type == 'hardsigm':
            return torch.clamp(x, 0, 1)
        elif type == 'tanh':
            return 0.5+0.5*torch.tanh(x)

    def forward(self, x):
        # TODO we can remove the args in the input variables
        if self.convNet:
            for i in range(self.conv_number):
                x = self.conv_number[i](x)
                x = F.relu(x),
                x = self.pool(x)

            # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            x = x.view(x.size(0), -1)

        for i in range(len(self.fcLayers)-1):
            x = self.rho(self.W[i](x), self.rho[i])

        return x
    # for i in range(len(args.fcLayers)-1):
    #         x = self.rho(self.W[i](x), args.rho[i])

        return output

    def defi_target_01(self, output, N, penalty_gate='NON'):

        '''Define the target also for the refactory period mechanism'''
        ''' The following function concerning the RP, which assumes that the batchSize = 1'''

        # define the target by the

        # create unsupervised target
        unsupervised_targets = torch.zeros(output.size())
        m_output = output.detach().clone()

        # Can RP be compatible with batch???
        if penalty_gate != 'NON':
            for i in range(output.size()[0]):
                for out in range(self.output):
                    # TODO here penalty gate is a vector considering only one image
                    if penalty_gate[out] == 1:
                        m_output[i, out] = -1

        # N_maxindex
        N_maxindex = torch.topk(m_output, N).indices

        for i in range(output.size()[0]):
            indices_0 = (m_output[i, :] == 0).nonzero(as_tuple=True)[0]
            indices_1 = (m_output[i, :] == 1).nonzero(as_tuple=True)[0]

            # if max(output) are 0
            if torch.max(m_output[i, :]) == 0 and indices_0.nelement() > N:
                random_idx0 = torch.multinomial(indices_0.type(torch.float), N).type(torch.long)
                N_maxindex[i, :] = indices_0[random_idx0]

            # if output have several 1
            elif indices_1.nelement() > N:
                random_idx1 = torch.multinomial(indices_1.type(torch.float), N).type(torch.long)
                N_maxindex[i, :] = indices_1[random_idx1]

        unsupervised_targets.scatter_(1, N_maxindex, torch.ones(output.size()))

        return unsupervised_targets, N_maxindex

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

    def defi_N_target(self, output, N):

        # define unsupervised target
        unsupervised_targets = torch.zeros(output.size(), device=self.device)

        # if self.cuda:
        #     unsupervised_targets = unsupervised_targets.to(self.device)
        #     src = src.to(self.device)

        # N_maxindex
        N_maxindex = torch.topk(output.detach(), N).indices  # N_maxindex has the size of (batch, N)

        # # !!! Attention if we consider the 0 and 1, which means we expect the values of input is between 0 and 1
        # # This probability occurs when we clamp the 'vector' between 0 and 1
        #
        # # Extract the batch where all the outputs are 0
        # # Consider the case where output is all 0 or 1
        # sum_output = torch.sum(m_output, axis=1)
        # indices_0 = (sum_output == 0).nonzero(as_tuple=True)[0]
        #
        # if indices_0.nelement() != 0:
        #     for i in range(indices_0.nelement()):
        #         N_maxindex[indices_0[i], :] = torch.randint(0, self.output-1, (N,))

        # unsupervised_targets[N_maxindex] = 1
        unsupervised_targets.scatter_(1, N_maxindex, torch.ones(output.size(), device=self.device))
        # print('the unsupervised vector is:', unsupervised_targets)

        return unsupervised_targets, N_maxindex

    '''The function below are for the homeostasis loss function'''

    def calculate_average_activity(self, output, Y_av):

        Y = self.eta*output + (1-self.eta)*Y_av

        return Y

    '''Add the weight normalization function'''
    def Weight_normal(self):
        with torch.no_grad():
            for i in range(len(self.fcLayers) - 1):
                # #H = torch.mean(self.W[i].weight.data, 1) / args.fcLayers[i]
                # H = torch.mean(self.W[i].weight.data, 1) / 10
                # #H = torch.mean(self.W[i].weight.data, 1)
                # H = H.expand(args.fcLayers[i], args.fcLayers[i+1])
                # self.W[i].weight.data = self.W[i].weight.data - H.t()
                for j in range(self.fcLayers[i + 1]):
                    self.W[i].weight[:, j].data = self.W[i].weight[:, j].data / torch.norm(self.W[i].weight[:, j].data)
                # self.W[i].weight.data = self.W[i].weight.data - (torch.mean(self.W[i].weight.data,1)/args.fcLayers[i+1]).expand(args.fcLayers[i+1], args.fcLayers[i]).t()
                # print('the layer is', i)
                # print('the weight after normalization is', self.W[i].weight.data)
                # print('the average is', torch.mean(self.W[i].weight.data,1))


class Classlayer(nn.Module):
    # one layer perceptron does not need to be trained by EP
    def __init__(self, jparams):
        super(Classlayer, self).__init__()
        # output_neuron=args.n_class
        self.output = jparams['n_class']
        self.input = jparams['fcLayers'][0]
        self.activation = jparams['class_activation']
        # define the classification layer
        self.class_layer = nn.Linear(self.input, self.output, bias=True)

        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)

    def rho(self, x, type):
        if type == 'relu':
            return F.relu(x)
        elif type == 'softmax':
            return F.softmax(x, dim=1)
        elif type == 'x':
            return x
        elif type == 'sigmoid':
            return torch.sigmoid(x)
        elif type == 'hardsigm':
            return torch.clamp(x, 0, 1)
        elif type == 'tanh':
            return 0.5+0.5*torch.tanh(x)

    def forward(self, x):
        x = self.rho(self.class_layer(x), self.activation)
        return x


