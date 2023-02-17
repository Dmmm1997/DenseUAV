import torch.optim as optim
from torch.optim import lr_scheduler


def make_optimizer(model,opt):
    ignored_params = []
    if opt.transformer:
        if opt.views==3:
            for i in [model.model_1, model.model_2, model.model_3]:
                ignored_params += list(map(id, i.transformer.parameters()))
        else:
            # for i in [model.model_1, model.model_2]:
            for i in [model.model_1]:
                ignored_params += list(map(id, i.transformer.parameters()))
        extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.3 * opt.lr},
            {'params': extra_params, 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    else:
        if opt.views==3:
            for i in [model.model_1,model.model_2,model.model_3]:
                ignored_params += list(map(id,i.model.parameters()))
        else:
            for i in [model.model_1]:
                ignored_params += list(map(id, i.model.parameters()))
        # ignored_params = list(map(id, model.classifier.parameters() ))
        extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        base_params = filter(lambda  p: id(p) in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': extra_params, 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[70,110], gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=4, verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

    return optimizer_ft,exp_lr_scheduler