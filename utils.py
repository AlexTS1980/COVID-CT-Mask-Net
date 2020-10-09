import torch
import torch.nn

# The Classification module must have the word 'new' in it
# set all trainable modules to training mode
# To replicate the best model from the paper, 
# the update mode must be heads_bn, but BatchNorm2d 
# layers must be in evaluation mode
# The weights will be updated, but means/variances frozen
def set_to_train_mode(model, update_mode):
    for _k in model._modules.keys():
        if 'new' in _k or update_mode == 'full':
            model._modules[_k].train(True)


# copy weights to existing layers, switch on gradients for other layers
# This doesn't apply to running_var, running_mean and batch tracking
# This assumes that the classifier layers has the 'new' in their name
# or 'bn' for BatchNorm
def switch_model_on(model, list_trained_pars, update_mode):
    for _n, _p in model.named_parameters():
        _p.requires_grad_(True)
        if 'new' in _n or update_mode == 'heads_bn' and 'bn' in _n or update_mode == "full":
            list_trained_pars.append(_p)
            print(_n, 'trainable parameters')


# easier to get booleans
def str_to_bool(s):
    return s.lower() in ('true')
