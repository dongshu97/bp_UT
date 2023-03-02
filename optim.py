import nevergrad as ng
import numpy as np
import torchvision
import argparse
import pathlib
from pathlib import Path
import pandas as pd
from Data import *
from Network import *
from Tools import *


def returnMNIST(batchSize,  conv, batchSizeTest=256):

    # define the optimization dataset
    print('We use the MNIST Dataset')
    # Define the Transform
    # !! Attention it depends on whether use the convolutional layers

    if conv == 0:
        transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]
    else:
        transforms = [torchvision.transforms.ToTensor()]

    # Down load the MNIST dataset
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                           transform=torchvision.transforms.Compose(transforms),
                                           target_transform=ReshapeTransformTarget(10))

    rest_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    validation_seed = 0


    validation_set = ValidationDataset(root='./MNIST_validate_seed', rest_set=rest_set, seed=validation_seed,
                             transform=torchvision.transforms.Compose(transforms))

    classValidation_set = ClassDataset(root='./MNIST_classValidate', test_set=validation_set, seed=validation_seed,
                             transform=torchvision.transforms.Compose(transforms))
    test_set = HypertestDataset(root='./MNIST_validate_seed', rest_set=rest_set, seed=validation_seed,
                             transform=torchvision.transforms.Compose(transforms))
    classTest_set = ClassDataset(root='./MNIST_classTest', test_set=test_set, seed=validation_seed,
                                       transform=torchvision.transforms.Compose(transforms))

    # load the datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batchSizeTest, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSizeTest, shuffle=True)
    classValidation_loader = torch.utils.data.DataLoader(classValidation_set, batch_size=300, shuffle=True)
    classTest_loader = torch.utils.data.DataLoader(classTest_set, batch_size=700, shuffle=True)

    return train_loader, validation_loader, test_loader, classValidation_loader, classTest_loader


def argsCreate(device, epoch, batchSize, lr, eta, T, gamma, nudge_N, convNet, layersList, rho, method='bp_Xth'):

    parser = argparse.ArgumentParser(description='usupervised EP')
    parser.add_argument(
        '--device',
        type=int,
        default=device,
        help='GPU name to use cuda')
    parser.add_argument(
        '--dataset',
        type=str,
        default="mnist",
        help='dataset to be used to train the network : (default = digits, other: mnist)')
    parser.add_argument(
        '--action',
        type=str,
        default=method,
        help='the type of bp test: (default = bp, other: bp_wta, bp_rp, test, visu')
    parser.add_argument(
        '--epochs',
        type=int,
        default=epoch,
        metavar='N',
        help='number of epochs to train (default: 50)')
    parser.add_argument(
        '--batchSize',
        type=int,
        default=batchSize,
        help='Batch size (default=1)')
    parser.add_argument(
        '--test_batchSize',
        type=int,
        default=256,
        help='Testing batch size (default=256)')
    parser.add_argument(
        '--ConvNET',
        type=int,
        default=convNet,
        help='Whether to use the ConvNet'
    )
    parser.add_argument(
        '--layersList',
        nargs='+',
        type=int,
        default=layersList,
        help='List of fully connected layers in the model')
    parser.add_argument(
        '--lr',
        nargs='+',
        type=float,
        default=lr,
        help='learning rate (default = 0.001)')
    parser.add_argument(
        '--rho',
        nargs='+',
        type=str,
        default=rho,
        help='define the activation function of each layer (it has to the str type)'
    )
    parser.add_argument(
        '--n_class',
        type=int,
        default=10,
        help='the number of the class (default = 10)')
    parser.add_argument(
        '--eta',
        type=float,
        default=eta,
        help='coefficient of homeostasis (default=0.6)')
    parser.add_argument(
        '--target_activity',
        type=float,
        default=T,
        help='coefficient of homeostasis (default=0.1)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=gamma,
        help='the multiplicative coefficient for Xth change'
    )
    parser.add_argument(
        '--nudge_N',
        type=int,
        default=nudge_N,
        help='the number of nudged neurons for each example'
    )
    parser.add_argument(
        '--WN',
        type=int,
        default=0,
        help='whether to use the weight normalization'
    )
    parser.add_argument(
        '--spike',
        type=int,
        default=0,
        help='whether we print out the spike results'
    )
    args = parser.parse_args()

    return args


def training(lr:float, alpha1:float, alpha2:float, alpha3:float, batchSize:int, gamma:float, eta:float, epoch:int, method:str, convNet:int, layersList:list, rho1:str, rho2:str, nudge_N:int):

    # We load the dataset
    train_loader, validation_loader, test_loader, \
    classValidation_loader, classTest_loader = returnMNIST(batchSize, conv=convNet)

    Target_activity = nudge_N/layersList[-1]

    args = argsCreate(device=0, epoch=epoch, batchSize=batchSize, lr=[lr*alpha1, lr*alpha2, lr*alpha3, lr],
                      eta=eta, T=Target_activity, gamma=gamma, nudge_N=nudge_N, convNet=convNet,
                      layersList=layersList, rho=[rho1, rho2], method=method)

    net = Net(args)

    if method == 'bp_Xth':

        validation_error_av = []
        validation_error_max = []
        for epoch in range(args.epochs):
            if epoch == 0:
                Xth = torch.zeros(args.layersList[-1])

            Xth = train_Xth(net, args, train_loader, epoch, Xth=Xth)

            # classifying process
            response = classify(net, args, classValidation_loader)

            # testing process
            error_av_epoch, error_max_epoch = test_Xth(net, args, validation_loader, response=response)
            validation_error_av.append(error_av_epoch.item())
            validation_error_max.append(error_max_epoch.item())

            one2one_av_min = min(validation_error_av)
            one2one_max_min = min(validation_error_max)

            # TODO to be change back to one2one av and max version
            if error_av_epoch > one2one_av_min*1.03 and error_max_epoch > one2one_max_min*1.03:
            #if error_max_epoch > one2one_max_min * 1.03:
                break

        response = classify(net, args, classTest_loader)
        test_error_av, test_error_max = test_Xth(net, args, test_loader, response=response)

        print('The training finish at Epoch:', epoch)
        return float(test_error_av), float(test_error_max), epoch

    elif method == 'bp':

        validation_train_error = []
        validation_test_error = []

        for epoch in range(args.epochs):

            train_error_epoch = train_bp(net, args, train_loader, epoch)

            # testing process
            test_error_epoch = test_bp(net, args, validation_loader)

            validation_train_error.append(train_error_epoch.item())
            validation_test_error.append(test_error_epoch.item())

            validation_train_min = min(validation_train_error)
            validation_test_min = min(validation_test_error)

            # TODO to be change back to one2one av and max version
            if train_error_epoch > validation_train_min * 1.02 and test_error_epoch > validation_test_min * 1.02:
                # if error_max_epoch > one2one_max_min * 1.03:
                break

        test_error_final = test_bp(net, args, test_loader)

        #test_error_av, test_error_max = test_Xth(net, args, test_loader, response=response)

        print('The training finish at Epoch:', epoch)
        return float(test_error_final), epoch


def createHyperFile(parameters):

    Path('./HyperTest').mkdir(parents=True, exist_ok=True)
    path_hyper = Path(Path.cwd()/'HyperTest')

    filePathList = list(path_hyper.glob('*.csv'))


    if len(filePathList) == 0:
        filename = 'H-1.csv'
        filePath = path_hyper/filename
    else:
        tab = []
        for i in range(len(filePathList)):
            tab.append(int(filePathList[i].stem.split('-')[-1]))
        filename = 'H-' + str(max(tab)+1) + '.csv'
        filePath = path_hyper/filename


    if Path(filePath).is_file():
        dataframe = pd.read_csv(filePath, sep=',', index_col=0)
    else:
        if parameters['method']=='bp_Xth':
            columns_header = ['lr', 'alpha1', 'alpha2', 'alpha3', 'batch', 'gamma', 'eta', 'rho1', 'rho2', 'final_epoch', 'one2one_av', 'one2one_max']
        elif parameters['method']=='bp':
            columns_header = ['lr', 'alpah1', 'alpha2', 'alpha3', 'batch', 'gamma', 'eta', 'rho1', 'rho2', 'final_epoch', 'test_error']

        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(filePath)

    return dataframe, filePath, path_hyper


def updataDataframe(filePath, dataframe, parameters, epoch, one2one_av='NON', one2one_max='NON', test_error='NON'):

    if parameters['method']=='bp_Xth':
        data = [parameters['lr'], parameters['alpha1'], parameters['alpha2'], parameters['alpha3'], parameters['batchSize'], parameters['gamma'], parameters['eta'], parameters['rho1'],  parameters['rho2'], epoch, one2one_av, one2one_max]
    elif parameters['method']=='bp':
        data = [parameters['lr'], parameters['alpha1'], parameters['alpha2'], parameters['alpha3'], parameters['batchSize'], parameters['gamma'], parameters['eta'], parameters['rho1'], parameters['rho2'], epoch, test_error]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(filePath)
    except PermissionError:
        input("Close the result.csv and press any key.")


    return dataframe


if __name__ == '__main__':

    parametrization = ng.p.Instrumentation(
        lr=ng.p.Log(lower=0.001, upper=1),
        alpha1=1,
        #alpha1=ng.p.Log(lower=0.1, upper=10),
        #alpha2=ng.p.Log(lower=0.1, upper=10),
        alpha2=1,
        #alpha3=ng.p.Scalar(lower=0.0, upper=1.0),
        alpha3=1,
        #batchSize=ng.p.Scalar(lower=10, upper=300).set_integer_casting(),
        batchSize=ng.p.Choice([64, 128, 256]),
        #gamma=ng.p.Log(lower=0.001, upper=1.0),
        gamma=0.5,
        eta=0.6,
        epoch=50,
        method='bp',
        convNet=0,
        #layersList=[32*7*7, 10],
        layersList=[784, 10],
        rho1=ng.p.Choice(["softmax", "clamp"]),
        #rho1='relu',
        #rho2=ng.p.Choice(["softmax", "clamp"]),
        rho2='clamp',
        nudge_N=1
    )
    #optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=50, num_workers=2)
    #TODO change the num_workers back to 2
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=50, num_workers=1)
    # optim.suggest(lr=0.0005, batch_size=40, gamma=0.1)


    #average_N = 1

    x = optimizer.ask()

    dataframe, filePath, path_hyper = createHyperFile(x.kwargs)

    method = x.kwargs['method']

    optimizedFile = 'initial' + Path(filePath).stem + '.txt'

    with open(path_hyper / optimizedFile, 'w') as f:
        for key, value in x.kwargs.items():
            f.write('%s:%s\n' % (key, value))
        f.write('%s:%s\n' % (key, value))

    if method == 'bp_Xth':

        for _ in range(optimizer.budget):
            x1 = optimizer.ask()
            x2 = optimizer.ask()

        #print('kwargs of x1 is', x1.kwargs)
        #print('kwargs of x2 is', x2.kwargs)
        # av_avError, av_maxError, av_epoch = 0
        # for av in range(average_N):
        #     one2one_av, one2one_max, epoch = training(*x.args, **x.kwargs)
        #     print('one2one_av is', one2one_av, 'one2one_max is', one2one_max)
        #     av_avError += one2one_av
        #     av_maxError += one2one_max
        #     av_epoch += epoch
        # av_avError = av_avError/average_N
        # av_maxError = av_maxError/average_N
        # av_epoch = av_epoch/average_N
            one2one_av1, one2one_max1, epoch1 = training(*x1.args, **x1.kwargs)
            one2one_av2, one2one_max2, epoch2 = training(*x2.args, **x2.kwargs)
            #print('one2one_av1 is', one2one_av1, 'one2one_max1 is', one2one_max1)
            #print('one2one_av2 is', one2one_av2, 'one2one_max2 is', one2one_max2)
            #TODO to be change back to one2one_av
            #optimizer.tell(x1, one2one_max1)
            #optimizer.tell(x2, one2one_max2)
            optimizer.tell(x1, one2one_av1)
            optimizer.tell(x2, one2one_av2)

            dataframe = updataDataframe(filePath, dataframe, x1.kwargs, epoch1, one2one_av=one2one_av1, one2one_max=one2one_max1)
            dataframe = updataDataframe(filePath, dataframe, x2.kwargs, epoch2, one2one_av=one2one_av2, one2one_max=one2one_max2)

    elif method == 'bp':

        for _ in range(optimizer.budget):
            x1 = optimizer.ask()
            x2 = optimizer.ask()
            test_error1, epoch1 = training(*x1.args, **x1.kwargs)
            test_error2, epoch2 = training(*x2.args, **x2.kwargs)
            dataframe = updataDataframe(filePath, dataframe, x1.kwargs, epoch1, test_error=test_error1)
            dataframe = updataDataframe(filePath, dataframe, x2.kwargs, epoch2, test_error=test_error2)



    recommendation = optimizer.recommend()
    print('The recommendation kwargs is', recommendation.kwargs)

    optimizedFile = 'optimized'+Path(filePath).stem + '.txt'
    with open(path_hyper/optimizedFile, 'w') as f:
        for key, value in recommendation.kwargs.items():
            f.write('%s:%s\n' % (key, value))
        f.write('%s:%s\n' % (key, value))
