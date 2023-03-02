import optuna
import torch
import numpy as np
import pandas
from tqdm import tqdm
import torchvision
import argparse
from pathlib import Path
from Data import *
from Network import *
from Tools import *

parser = argparse.ArgumentParser(description='hyperparameter EP by optuna')
parser.add_argument(
    '--device',
    type=int,
    default=0,
    help='GPU name to use cuda')
parser.add_argument(
    '--dataset',
    type=str,
    default='YinYang',
    help='Dataset (default:YinYang, else:mnist)'
)
parser.add_argument(
    '--action',
    type=str,
    default='bp_Xth',
    help='Decide the learning method (default:bp, else:bp_Xth)'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    metavar='N',
    help='number of epochs to train (default: 200)')
parser.add_argument(
    '--test_batchSize',
    type=int,
    default=128,
    help='Testing batch size (default=256)')
parser.add_argument(
    '--ConvNET',
    type=int,
    default=0,
    help='Whether to use the ConvNet'
)
parser.add_argument(
    '--convLayers',
    nargs='+',
    type=int,
    default=[1, 32, 5, 1, 0, 32, 64, 5, 1, 0],
    help='The parameters of convNet, each conv layer has 5 parameter: in_channels, out_channels, K/F, S, P')
parser.add_argument(
    '--structure',
    nargs='+',
    type=int,
    default=[64*7*7, 10],
    help='Test structure')
parser.add_argument(
    '--rho',
    nargs='+',
    type=str,
    default=['relu', 'clamp'],
    help='define the activation function of each layer (it has to the str type)'
)
# parser.add_argument(
#     '--loss',
#     type=str,
#     default='Cross-entropy',
#     help='define the loss function that we used (default: MSE, else:Cross-entropy)'
# )
parser.add_argument(
    '--n_class',
    type=int,
    default=10,
    help='the number of class (default = 10)'
)
parser.add_argument(
    '--Homeo_mode',
    type=str,
    default='batch',
    help='batch mode or SM mode'
)
parser.add_argument(
    '--exp_N',
    type=int,
    default=1,
    help='N winner (default: 1)')
# parser.add_argument(
#     '--Optimizer',
#     type=str,
#     default='SGD',
#     help='the optimizer to be used (default=SGD, else:Adam)'
# )
exp_args = parser.parse_args()


# define the dataset
def returnMNIST(batchSize,batchSizeTest=256):
    # TODO adapt the dataset for the CNN
    # TODO delete the validation set and replace by the test dataset

    # define the optimization dataset
    print('We use the MNIST Dataset')
    # Define the Transform
    # !! Attention it depends on whether use the convolutional layers

    transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

    # Down load the MNIST dataset
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                           transform=torchvision.transforms.Compose(transforms),
                                           target_transform=ReshapeTransformTarget(10))

    rest_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    validation_seed = 1

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


def returnYinYang(batchSize, batchSizeTest=128):
    print('We use the YinYang dataset')
    if exp_args.action == 'bp':
        train_set = YinYangDataset(size=5000, seed=42, target_transform=ReshapeTransformTarget(3))
    elif exp_args.action == 'bp_Xth':
        train_set = YinYangDataset(size=5000, seed=42)

    validation_set = YinYangDataset(size=1000, seed=41)  # used for the hyperparameter research
    classValidation_set = YinYangDataset(size=1000, seed=41, sub_class=True)

    test_set = YinYangDataset(size=1000, seed=40)
    classTest_set = YinYangDataset(size=1000, seed=40, sub_class=True)

    # seperate the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSizeTest, shuffle=False)

    classValidation_loader = torch.utils.data.DataLoader(classValidation_set, batch_size=100, shuffle=False)
    classTest_loader = torch.utils.data.DataLoader(classTest_set, batch_size=100, shuffle=False)

    return train_loader, validation_loader, test_loader, classValidation_loader, classTest_loader


def argsCreate(exp_args, batchSize, lr, loss, eta, gamma, nudge_N):
    args = argparse.Namespace()
    args.device = exp_args.device
    args.dataset = exp_args.dataset
    args.action = exp_args.action
    args.epochs = exp_args.epochs
    args.batchSize = batchSize
    args.test_batchSize = exp_args.test_batchSize
    args.ConvNET = 0
    args.convLayers = [1, 16, 5, 1, 1, 16, 32, 5, 1, 1]
    args.layersList = exp_args.structure.copy()
    args.lr = lr.copy()
    args.rho = exp_args.rho
    args.loss = loss
    args.n_class = exp_args.n_class
    args.eta = eta
    args.gamma = gamma
    args.nudge_N = nudge_N
    return args


def train_validation_test(args, net, trial, train_loader, validation_loader, test_loader, classValidation_loader, classTest_loader):

    # train the model
    if exp_args.action == 'bp':
        print("Training the model with supervised ep")

        for epoch in tqdm(range(args.epochs)):
            train_error_epoch = train_bp(net, args, train_loader, epoch)
            validation_error_epoch = test_bp(net, args, validation_loader)

            # Handle pruning based on the intermediate value.
            trial.report(validation_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        test_error = test_bp(net, args, validation_loader)
        return test_error

    elif args.action == 'bp_Xth':
        print("Training the model with unsupervised ep")

        for epoch in tqdm(range(args.epochs)):
            # train process
            Xth = train_Xth(net, args, train_loader, epoch)
            # class process
            response = classify(net, args, classValidation_loader)
            # test process
            error_av_epoch, error_max_epoch = test_Xth(net, args, validation_loader, response=response,
                                                                 spike_record=0)

            # Handle pruning based on the intermediate value.
            trial.report(error_av_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        response = classify(net, args, classValidation_loader)
        error_av, error_max = test_Xth(net, args, validation_loader, response=response,
                                                                 spike_record=0)
        return error_av


def objective(trial, exp_args):

    # design the hyperparameters to be optimized
    loss = trial.suggest_categorical("loss", ["MSE", "Cross-entropy"])
    lr1 = trial.suggest_float("lr1", 1e-5, 0.1, log=True)
    lr_coeff = trial.suggest_float("lr_coeff", 0.5, 4)
    lr = [lr_coeff*lr1, lr1]

    if exp_args.action == 'bp':
        eta = 0.6
        gamma = 0.8
        nudge_N = 1
    else:
        if exp_args.Homeo_mode == 'batch':
            batchSize = trial.suggest_int("batchSize", 10, 128)
            eta = 0.6
        else:
            batchSize = 1
            eta = trial.suggest_float("eta", 0.001, 1, log=True)
        gamma = trial.suggest_float("gamma", 0.001, 1, log=True)
        nudge_N = trial.suggest_int("nudge_N", 1, exp_args.exp_N)

    # create the args for the training'
    args = argsCreate(exp_args, batchSize, lr, loss, eta, gamma, nudge_N)
    # args.fcLayers.reverse()  # we put in the other side, output first, input last
    # args.lr.reverse()
    # args.dropProb.reverse()

    # create the dataset
    if exp_args.dataset == 'YinYang':
        train_loader, validation_loader, test_loader, classValidation_loader, classTest_loader=\
            returnYinYang(batchSize, batchSizeTest=exp_args.test_batchSize)
    elif exp_args.dataset == 'MNIST':
        train_loader, validation_loader, test_loader, classValidation_loader, classTest_loader=\
            returnMNIST(batchSize, batchSizeTest=exp_args.test_batchSize)

    # create the model
    net = Net(args)
    # training process
    final_err = train_validation_test(args, net, trial, train_loader, validation_loader, test_loader, classValidation_loader,
                          classTest_loader)

    return final_err


def saveHyperparameters(exp_args, BASE_PATH):
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

    for key in exp_args.__dict__:
        f.write(key)
        f.write(': ')
        f.write(str(exp_args.__dict__[key]))
        f.write('\n')

    f.close()


def optuna_createPath():
    '''
    Create path to save data
    '''
    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    BASE_PATH +=  prefix + 'Optuna-0'

    # BASE_PATH += prefix + args.dataset
    #
    # BASE_PATH += prefix + 'method-' + args.method
    #
    # BASE_PATH += prefix + args.action
    #
    # BASE_PATH += prefix + str(len(args.fcLayers)-2) + 'hidden'
    # BASE_PATH += prefix + 'hidNeu' + str(args.layersList[1])
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


if __name__=='__main__':
    # define prefix
    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    # define Sampler

    # define Pruner

    # define the dataframe
    BASE_PATH, name = optuna_createPath()
    # save hyperparameters
    saveHyperparameters(exp_args, BASE_PATH)
    # create the filepath for saving the optuna trails
    filePath = BASE_PATH + prefix + "test.csv"
    study_name = str(time.asctime())
    study = optuna.create_study(study_name=study_name, storage='sqlite:///optuna_bp_unsupervised.db')
    study.optimize(lambda trial: objective(trial, exp_args), n_trials=100)
    trails = study.get_trials()
    # record trials
    df = study.trials_dataframe()
    df.to_csv(filePath)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_slice(study)


    #np.savetxt(BASE_PATH + prefix + "test.csv", trails, delimiter=",", fmt='%s')
    #np.savetxt(BASE_PATH+"test.csv", trails, delimiter=",", fmt='%s', header=header)
    # save study and read the parameters in the study








