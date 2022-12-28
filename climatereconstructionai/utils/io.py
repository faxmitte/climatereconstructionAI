import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, device, optimizers=None):
    ckpt_dict = torch.load(ckpt_name, map_location=device)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        ckpt_dict[prefix] = {key.replace("module.", ""): value for key, value in ckpt_dict[prefix].items()}
        model.load_state_dict(ckpt_dict[prefix], strict=False)

    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


def load_model(ckpt_dict, model, optimizer=None, label=None):
    assert isinstance(model, nn.Module)
    if label is None:
        label = ckpt_dict["labels"][-1]

    ckpt_dict[label]["model"] = \
        {key.replace("module.", ""): value for key, value in ckpt_dict[label]["model"].items()}
    model.load_state_dict(ckpt_dict[label]["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt_dict[label]["optimizer"])
    return ckpt_dict[label]["n_iter"]