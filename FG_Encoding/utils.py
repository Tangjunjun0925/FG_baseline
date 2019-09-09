def set_parameter_requires_grad(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad


def gather_the_parameters(model_ft):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    # params_to_update = model_ft.parameters()
    print("Params to learn:")
    # if feature_extract:
    params_to_update = []
    params_no_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
        else:
            params_no_update.append(param)
    # else:
    #     for name, param in model_ft.named_parameters():
    #         # param_lr = [{'params': model_ft.conv1.parameters(), 'lr': 0.2},
    #         # {'params': model_ft.conv2.parameters(), 'lr': 0.2},
    #         # {'params': prelu_params, 'lr': 0.02},
    #         # {'params': rest_params, 'lr': 0.3}
    #         # ]
    #         if param.requires_grad == True:
    #             print("\t", name)
    #return params_to_update
