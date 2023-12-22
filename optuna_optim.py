
import pandas
import argparse
import json
import copy as CP
from pathlib import Path
from Data import *
from actions import *
import logging
import sys

# load the parameters in optuna_config

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'.',
    #default=r'D:\Results_data\BP_perceptron_without_dropout\784-100',
    help='path of json configuration'
)
parser.add_argument(
    '--trained_path',
    type=str,
    # default=r'.',
    default=r'.\pretrain_file',
    help='path of model_dict_state_file'
)

args = parser.parse_args()

# load the parameters in optuna_config
with open(args.json_path +prefix + 'optuna_config.json') as f:
  pre_config = json.load(f)

def returnYinYang(batchSize, batchSizeTest=128):
    print('We use the YinYang dataset')

    train_set = YinYangDataset(size=5000, seed=42, target_transform=ReshapeTransformTarget(3))
    validation_set = YinYangDataset(size=1000, seed=41)  # used for the hyperparameter research
    class_set = YinYangDataset(size=1000, seed=42, sub_class=True)
    layer_set = YinYangDataset(size=1000, seed=42, target_transform=ReshapeTransformTarget(3), sub_class=True)

    # test_set = YinYangDataset(size=1000, seed=40)
    # seperate the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batchSizeTest, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSizeTest, shuffle=False)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=100, shuffle=True)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=100, shuffle=True)

    return train_loader, validation_loader, class_loader, layer_loader


def jparamsCreate(pre_config, trial):

    jparams = CP.deepcopy(pre_config)

    # if jparams["dataset"] == 'mnist':
    #     jparams["class_seed"] = trial.suggest_int("class_seed", 0, 42)
        # else:
        #     jparams["pre_batchSize"] = trial.suggest_int("pre_batchSize", 10, min(jparams["trainLabel_number"], 512))

    if jparams["action"] == 'bp_Xth':
        if jparams['Homeo_mode'] == 'batch':
            # jparams["batchSize"] = trial.suggest_int("batchSize", 10, 256)
            jparams["batchSize"] = trial.suggest_categorical("batchSize", [16, 32, 64, 128])
            jparams["eta"] = 0.5
        else:
            jparams["batchSize"] = 1,
            jparams["eta"] = trial.suggest_float("eta", 0.01, 1, log=True)

        jparams["gamma"] = trial.suggest_float("gamma", 0.01, 0.9, log=True)
        jparams["pre_batchSize"] = 32
        # jparams["gamma"] = 0.2
        jparams["nudge_N"] = trial.suggest_int("nudge_N", 1, jparams["nudge_max"])

        lr = []
        if jparams["convNet"]:
            for i in range(len(jparams["C_list"])-1):
                lr_i = trial.suggest_float("lr"+str(i), 1e-6, 1, log=True)
                lr.append(lr_i)
            for i in range(len(jparams["fcLayers"])):
                lr_i = trial.suggest_float("lr" + str(i+len(jparams["C_list"])-1), 1e-5, 10, log=True)
                # to verify whether we need to change the name of lr_i
                lr.append(lr_i)

        else:
            for i in range(len(jparams["fcLayers"])-1):
                lr_i = trial.suggest_float("lr"+str(i), 1e-5, 10, log=True)
                # to verify whether we need to change the name of lr_i
                lr.append(lr_i)
        jparams["lr"] = lr.copy()

        jparams['factor'] = trial.suggest_float("factor", 1e-4, 1e-2, log=True)
        jparams['scheduler_epoch'] = trial.suggest_int("scheduler_epoch", 1, jparams['epochs'])
        jparams["scheduler"] = 'linear'
        # jparams["scheduler"] = trial.suggest_categorical("scheduler",
        #                                                      ['linear', 'step', 'cosine'])
        # jparams["exp_factor"] = trial.suggest_float("pre_gamma", 0.9, 0.999, log=True)
        jparams["exp_factor"] = 0.5

        if jparams['Prune'] == "Initialization":
            pruneAmount = []
            for i in range(len(lr)):
                PA_i = trial.suggest_float("PA" + str(i), 0, 1, log=False)
                #PA_i = 0.99
                pruneAmount.append(PA_i)
            jparams["pruneAmount"] = pruneAmount.copy()

    elif jparams["action"] == 'bp':

        jparams["batchSize"] = trial.suggest_int("batchSize", 10, 256)
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["nudge_N"] = 1

        lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("lr" + str(i), 1e-9, 1, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()

        jparams["Optimizer"] = 'Adam'
        #jparams["Optimizer"] = trial.suggest_categorical("Optimizer", ['SGD', 'Adam'])
        if jparams["Dropout"]:
            dropProb = [0.2, 0.5, 0]
            jparams["dropProb"] = dropProb.copy()
            # jparams["dropProb"].reverse()

    elif jparams["action"] == 'semi_supervised':
        # unchanged
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.2
        jparams["nudge_N"] = 1
        jparams["pre_batchSize"] = 32
        jparams["dropProb"] = [0.4, 0.5, 0]
        jparams["Optimizer"] = 'Adam'
        jparams["batchSize"] = 128
        # test
        lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("lr" + str(i), 1e-4, 1e-2, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams["supervised_start"] = trial.suggest_float("supervised_start", 1e-3, 1e-1, log=True)
        jparams["supervised_end"] = trial.suggest_float("supervised_end", 1e-3, 1e-1, log=True)
        jparams["unsupervised_start"] = trial.suggest_float("unsupervised_start", 1e-4, 1e-1, log=True)
        jparams["unsupervised_end"] = trial.suggest_float("unsupervised_end", 1e-4, 1e-1, log=True)

    elif jparams["action"] == 'pre_supervised_bp':
        jparams["batchSize"] = 256
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.3
        jparams["pre_batchSize"] = trial.suggest_int("pre_batchSize", 16, 128)
        pre_lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("pre_lr" + str(i), 1e-7, 1e-1, log=True)
            # to verify whether we need to change the name of lr_i
            pre_lr.append(lr_i)
        jparams["pre_lr"] = pre_lr.copy()
        jparams["Optimizer"] = 'Adam'
        if jparams["Dropout"]:
            dropProb = [0.2, 0.5, 0]
            jparams["dropProb"] = dropProb.copy()
        jparams["pre_scheduler"] = trial.suggest_categorical("pre_scheduler", ['linear', 'exponential', 'step', 'cosine'])
        jparams["pre_scheduler_epoch"] = trial.suggest_int("pre_scheduler_epoch", 1, jparams['pre_epochs'])
        jparams["pre_factor"] = trial.suggest_float("pre_factor", 1e-5, 1, log=True)
        jparams["pre_exp_factor"] = trial.suggest_float("pre_exp_factor", 0.9, 0.999, log=True)

    elif jparams["action"] == 'train_class_layer':
        # non-used parameters
        jparams["batchSize"] = 256
        jparams["pre_batchSize"] = 32
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["lr"] = [6, 3.35, 2.97, 1.83]
        jparams['Optimizer'] = 'SGD'
        # unchanged parameters
        jparams["class_Optimizer"] = 'Adam'
        # class
        jparams["class_activation"] = trial.suggest_categorical("class_activation", ['sigmoid', 'x', 'hardsigm'])
        jparams["class_lr"] = [trial.suggest_float("class_lr", 1e-4, 10, log=True)]
        jparams["class_scheduler"] = trial.suggest_categorical("class_scheduler",
                                                             ['linear', 'exponential'])
        jparams["class_scheduler_epoch"] = trial.suggest_int("class_scheduler_epoch", 1, jparams['class_epoch'])
        jparams["class_factor"] = trial.suggest_float("class_factor", 1e-5, 1, log=True)
        jparams["class_exp_factor"] = trial.suggest_float("pre_exp_factor", 0.9, 0.999, log=True)

    return jparams


def objective(trial, pre_config):

    # design the hyperparameters to be optimized
    jparams = jparamsCreate(pre_config, trial)

    # create the dataset
    if jparams["dataset"] == 'YinYang':
        train_loader, validation_loader,  class_loader, layer_loader =\
            returnYinYang(jparams['batchSize'], batchSizeTest=jparams["test_batchSize"])
    elif jparams["dataset"] == 'mnist':
        print('We use the MNIST Dataset')
        (train_loader, test_loader,
         class_loader, layer_loader,
         supervised_loader, unsupervised_loader) = returnMNIST(jparams, validation=True)

    elif jparams["dataset"] == 'cifar10':
        print('We use the CIFAR10 dataset')
        (train_loader, test_loader,
         class_loader, layer_loader,
         supervised_loader, unsupervised_loader) = returnCifar10(jparams, validation=True)

    # create the model
    if jparams['convNet']:
        net = CNN(jparams)
    else:
        net = MLP(jparams)

    # load the trained unsupervised network when we train classification layer
    if jparams["action"] == 'train_class_layer':
        print("Train the supplementary class layer for unsupervised learning")
        trained_path = args.trained_path + prefix + 'model_state_dict.pt'
        final_err = train_class_layer(net, jparams, layer_loader, test_loader, trained_path=trained_path, trial=trial)

    elif jparams["action"] == 'bp_Xth':
        final_err = unsupervised_bp(net, jparams, train_loader, class_loader, test_loader, layer_loader, BASE_PATH=None, trial=trial)

    elif jparams["action"] == 'bp':
        final_err = supervised_bp(net, jparams, train_loader, test_loader, BASE_PATH=None, trial=trial)

    elif jparams["action"] == 'semi_supervised':
        trained_path = args.trained_path + prefix + 'model_pre_supervised_state_dict.pt'
        final_err = semi_supervised_bp(net, jparams, supervised_loader, unsupervised_loader, test_loader,
                           trained_path=trained_path,  trial=trial)

    elif jparams["action"] == 'pre_supervised_bp':
        final_err = pre_supervised_bp(net, jparams, supervised_loader, test_loader, BASE_PATH=None, trial=trial)

    df = study.trials_dataframe()
    df.to_csv(filePath)
    del(jparams)

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

    BASE_PATH += prefix + 'Optuna-0'

    BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    print("len(BASE_PATH)="+str(len(BASE_PATH)))


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

    # save the optuna configuration
    with open(BASE_PATH + prefix + "optuna_config.json", "w") as outfile:
        json.dump(pre_config, outfile)

    # create the filepath for saving the optuna trails
    filePath = BASE_PATH + prefix + "test.csv"
    study_name = str(time.asctime())
    if pre_config['action'] == 'bp_Xth':
        search_space = {
            'batchSize': [16, 32, 64],
            'factor': [0.1, 0.2, 0.4, 0.5],
            'scheduler_epoch': [20, 30, 40, 50],
            'lr0': [2.5, 3, 4, 5, 6, 7, 8, 9],
            'nudge_N': [7, 8, 9, 10],
            'gamma': [0.2, 0.3, 0.4, 0.5]
        }
    elif pre_config['action'] == 'pre_supervised_bp':
        search_space = {
            'pre_batchSize': [16, 32, 128],
            'pre_lr0': [0.0002, 0.0005, 0.0008, 0.001, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015],
            'pre_lr1': [0.0002, 0.0005, 0.0008, 0.001, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015],
            'pre_scheduler_epoch': [50, 100, 120, 150, 200],
            'pre_factor': [0.5, 0.2, 0.1, 0.05, 0.01, 0.02, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005],
            # 'pre_gamma': [0.9, 0.92, 0.95, 0.96, 0.98, 0.99, 0.992, 0.995, 0.998, 0.999],
        }

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(),
                                pruner=optuna.pruners.PercentilePruner(30, n_startup_trials=2, n_warmup_steps=3),
                                study_name=study_name, storage='sqlite:///optuna_bp_Conv.db')
    # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(search_space),
    #                             pruner=optuna.pruners.PercentilePruner(45, n_startup_trials=3, n_warmup_steps=5),
    #                             study_name=study_name, storage='sqlite:///optuna_bp_Conv.db')


    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(lambda trial: objective(trial, pre_config), n_trials=400)
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








