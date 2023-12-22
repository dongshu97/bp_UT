import os
import optuna
from Tools import *
from tqdm import tqdm

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'


def defineOptimizer(net, lr, type, momentum=0, dampening=0):

    # define Optimizer
    layer_names = []
    for idx, (name, param) in enumerate(net.named_parameters()):
        layer_names.append(name)
    parameters = []

    for idx, name in enumerate(layer_names):
        # update learning rate
        if idx % 2 == 0:
            lr_indx = int(idx / 2)
            lr_layer = lr[lr_indx]
        # append layer parameters
        parameters += [{'params': [p for n, p in net.named_parameters() if n == name and p.requires_grad],
                        'lr': lr_layer}]

    # construct the optimizer
    # TODO changer optimizer to ADAM
    if type == 'SGD':
        optimizer = torch.optim.SGD(parameters, momentum=momentum, dampening=dampening)
    elif type == 'Adam':
        optimizer = torch.optim.Adam(parameters)
    return parameters, optimizer


def update_momentum(optimizer, epoch, start_factor, end_factor):
    if epoch < 500:
        factor = (epoch/500)*end_factor + (1-epoch/500)*start_factor
    else:
        factor = end_factor

    optimizer.momentum = factor
    optimizer.dampening = factor


def defineScheduler(optimizer, type, decay_factor, decay_epoch, exponential_factor):
    # linear
    if type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                                           end_factor=decay_factor,
                                                           total_iters=decay_epoch)
    # exponential
    elif type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exponential_factor)
    # step
    elif type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=decay_factor)
    # combine cosine
    elif type == 'cosine':
        scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=decay_factor,
                                                         total_iters=decay_epoch)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epoch)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

    return scheduler


def pre_supervised_bp(net, jparams, supervised_loader, test_loader, BASE_PATH=None, trial=None):
    # define the pre-supervised training optimizer
    pretrain_params, pretrain_optimizer = defineOptimizer(net, jparams['pre_lr'], jparams['Optimizer'])

    # TODO try different type of scheduler
    # pretrain_scheduler = torch.optim.lr_scheduler.LinearLR(pretrain_optimizer, start_factor=1,
    #                                                        end_factor=jparams['pre_factor'],
    #                                                        total_iters=jparams['pre_scheduler_epoch'])
    pretrain_scheduler = defineScheduler(pretrain_optimizer, jparams['pre_scheduler'], jparams['pre_factor'],
                                         jparams['pre_scheduler_epoch'], jparams['pre_exp_factor'])

    # save the initial network
    if BASE_PATH is not None:
        torch.save(net.state_dict(), BASE_PATH + prefix + 'model_pre_supervised_state_dict0.pt')
        # init Dataframe
        PretrainFrame = initDataframe(BASE_PATH, method='bp', dataframe_to_init='pre_supervised.csv')
        # list to save error
        pretrain_error_list = []
        pretest_error_list = []

    # training process
    for epoch in tqdm(range(jparams['pre_epochs'])):
        if jparams['Optimizer'] == 'SGD':
            update_momentum(pretrain_optimizer, epoch, 0.5, 0.99)
        pretrain_error_epoch = train_bp(net, jparams, supervised_loader, epoch, pretrain_optimizer)
        # testing process
        pretest_error_epoch = test_bp(net, test_loader)
        pretrain_scheduler.step()

        if trial is not None:
            trial.report(pretest_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            pretrain_error_list.append(pretrain_error_epoch.item())
            pretest_error_list.append(pretest_error_epoch.item())
            # write the error in csv
            PretrainFrame = updateDataframe(BASE_PATH, PretrainFrame, pretrain_error_list, pretest_error_list,
                                        'pre_supervised.csv')
            # save the entire model
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_pre_supervised_state_dict.pt')

    if trial is not None:
        return pretest_error_epoch


def supervised_bp(net, jparams, train_loader, test_loader, BASE_PATH=None, trial=None):

    # define the optimizer
    params, optimizer = defineOptimizer(net, jparams['lr'], jparams['Optimizer'])

    # TODO test the optimizer on supervised bp
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if BASE_PATH is not None:
        DATAFRAME = initDataframe(BASE_PATH, method='bp')
        print(DATAFRAME)
        train_error = []
        test_error = []
    for epoch in tqdm(range(jparams['epochs'])):
        train_error_epoch = train_bp(net, jparams, train_loader, epoch, optimizer)
        # testing process
        test_error_epoch = test_bp(net, test_loader)
        # scheduler.step()

        if trial is not None:
            trial.report(test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            train_error.append(train_error_epoch.item())
            test_error.append(test_error_epoch.item())
            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, train_error, test_error)
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')
    if trial is not None:
        return test_error_epoch


def unsupervised_bp(net, jparams, train_loader, class_loader, test_loader, layer_loader, BASE_PATH=None, trial=None):
    # TODO to change it for the other uses

    # define optimizer
    params, optimizer = defineOptimizer(net, jparams['lr'], jparams['Optimizer'])

    # define scheduler
    # scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=jparams["factor"],
    #                                                  total_iters=jparams["scheduler_epoch"])
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=jparams["scheduler_epoch"])
    #
    # # scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.5, max_lr=1)
    # scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

    scheduler = defineScheduler(optimizer, jparams['scheduler'], jparams['factor'],
                                jparams['scheduler_epoch'], jparams['exp_factor'])

    if BASE_PATH is not None:
        DATAFRAME = initDataframe(BASE_PATH, method='bp_Xth')
        print(DATAFRAME)
        # dataframe for Xth
        Xth_dataframe = initXthframe(BASE_PATH, 'Xth_norm.csv')

        test_error_av = []
        test_error_max = []
        X_th = []

    for epoch in tqdm(range(jparams['epochs'])):
        Xth = train_Xth(net, jparams, train_loader, epoch, optimizer)

        # classifying process
        response = classify(net, jparams, class_loader)
        # testing process
        test_error_av_epoch, test_error_max_epoch = test_Xth(net, jparams, test_loader, response=response,
                                                             spike_record=0)
        # print('The test av error is:', test_error_av_epoch)
        scheduler.step()

        if trial is not None:
            trial.report(test_error_av_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            test_error_av.append(test_error_av_epoch.item())
            test_error_max.append(test_error_max_epoch.item())
            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, test_error_av, test_error_max)

            X_th.append(torch.norm(Xth).item())
            Xth_dataframe = updateXthframe(BASE_PATH, Xth_dataframe, X_th)
            # at each epoch, we update the model parameters
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')

    if trial is not None:
        return test_error_av_epoch

    # final direct association
    response = classify(net, jparams, class_loader)
    # testing process
    test_error_av_epoch, test_error_max_epoch = test_Xth(net, jparams, test_loader, response=response,
                                                         spike_record=0)
    if jparams['epochs'] == 0 and BASE_PATH is not None:
        test_error_av.append(test_error_av_epoch.item())
        test_error_max.append(test_error_max_epoch.item())
        DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, test_error_av, test_error_max)
    # train linear classifer
    train_class_layer(net, jparams, layer_loader, test_loader, BASE_PATH=BASE_PATH)


def train_class_layer(net, jparams, layer_loader, test_loader, trained_path=None, BASE_PATH=None, trial=None):
    # load the pre-trianed network
    if trained_path is not None:
        net.load_state_dict(torch.load(trained_path))

    # create the classification layer
    class_net = Classifier(jparams)

    # define optimizer
    class_params, class_optimizer = defineOptimizer(class_net, jparams['class_lr'], jparams['class_Optimizer'])

    # define scheduler
    class_scheduler = defineScheduler(class_optimizer, jparams['class_scheduler'], jparams['class_factor'],
                                jparams['class_scheduler_epoch'], jparams['class_exp_factor'])

    if BASE_PATH is not None:
        # create dataframe for classification layer
        class_dataframe = initDataframe(BASE_PATH, method='classification_layer',
                                        dataframe_to_init='classification_layer.csv')
        torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict_0.pt')
        class_train_error_list = []
        final_test_error_list = []
        final_loss_error_list = []

    for epoch in tqdm(range(jparams['class_epoch'])):
        # train
        class_train_error_epoch = classify_network(net, class_net, layer_loader, class_optimizer, jparams['class_smooth'])
        # test
        final_test_error_epoch, final_loss_epoch = test_unsupervised_layer(net, class_net, jparams, test_loader)
        # scheduler
        class_scheduler.step()

        if trial is not None:
            trial.report(final_test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            class_train_error_list.append(class_train_error_epoch.item())
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = updateDataframe(BASE_PATH, class_dataframe, class_train_error_list, final_test_error_list,
                                              filename='classification_layer.csv', loss=final_loss_error_list)
            # save the trained class_net
            torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict.pt')

    if trial is not None:
        return final_test_error_epoch


def semi_supervised_bp(net, jparams, supervised_loader, unsupervised_loader, test_loader,
                       trained_path=None, BASE_PATH=None, trial=None):

    if trained_path is not None:
        net.load_state_dict(torch.load(trained_path))
    else:
        # pretrain
        pre_supervised_bp(net, jparams, supervised_loader, test_loader, BASE_PATH=BASE_PATH)

    # read the initial pretrained error
    initial_pretrain_err = test_bp(net, test_loader)


    # define the supervised and unsupervised optimizer
    unsupervised_params, unsupervised_optimizer = defineOptimizer(net, jparams['lr'], jparams['Optimizer'])
    supervised_params, supervised_optimizer = defineOptimizer(net, jparams['lr'], jparams['Optimizer'])

    # define the supervised and unsupervised scheduler
    unsupervised_scheduler = torch.optim.lr_scheduler.LinearLR(unsupervised_optimizer,
                                                               start_factor=jparams['unsupervised_start'],
                                                               end_factor=jparams['unsupervised_end'], total_iters=jparams['epochs']-100)
    # TODO to be verify
    supervised_scheduler = torch.optim.lr_scheduler.LinearLR(supervised_optimizer,
                                                               start_factor=jparams['supervised_start'],
                                                             end_factor=jparams['supervised_end'], total_iters=jparams['epochs']-100)

    if BASE_PATH is not None:
        # init the semi-supervised frame
        SEMIFRAME = initDataframe(BASE_PATH, method='semi_supervised', dataframe_to_init='semi_supervised.csv')

        supervised_test_error_list = []
        entire_test_error_list = []

    for epoch in tqdm(range(jparams['epochs'])):
        # unsupervised training
        Xth = train_Xth(net, jparams, unsupervised_loader, epoch, unsupervised_optimizer)
        entire_test_epoch = test_bp(net, test_loader)
        unsupervised_scheduler.step()

        # supervised reminder
        supervised_train_epoch = train_bp(net, jparams, supervised_loader, epoch, supervised_optimizer)
        supervised_test_epoch = test_bp(net, test_loader)
        supervised_scheduler.step()

        # print('semi-supervised trained error is:', supervised_train_epoch)

        if trial is not None:
            trial.report(supervised_test_epoch, epoch)
            if entire_test_epoch > initial_pretrain_err:
                raise optuna.TrialPruned()
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            entire_test_error_list.append(entire_test_epoch.item())
            supervised_test_error_list.append(supervised_test_epoch.item())
            SEMIFRAME = updateDataframe(BASE_PATH, SEMIFRAME, entire_test_error_list, supervised_test_error_list,
                                    'semi_supervised.csv')
            # save the entire model
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_semi_state_dict.pt')

    if trial is not None:
        return supervised_test_epoch