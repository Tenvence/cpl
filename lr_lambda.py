from bisect import bisect_right


def get_warm_up_lr_lambda(iters, ratio):
    return lambda epoch: (1. - ratio) / (iters - 1.) * (epoch + 1.) + (iters * ratio - 1.) / (iters - 1.)


def get_multi_step_lr_lambda(milestones, gamma=0.1):
    return lambda epoch: gamma ** bisect_right(milestones, epoch)


def get_warm_up_multi_step_lr_lambda(iters_per_epoch, warm_up_epoch, warm_up_ratio, milestones, gamma=0.1):
    warm_up_lr_lambda = get_warm_up_lr_lambda(warm_up_epoch * iters_per_epoch, warm_up_ratio)
    multi_step_lr_lambda = get_multi_step_lr_lambda(milestones, gamma)
    return lambda it: warm_up_lr_lambda(it) if it // iters_per_epoch < warm_up_epoch else multi_step_lr_lambda(it // iters_per_epoch)
