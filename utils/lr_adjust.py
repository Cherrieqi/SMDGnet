import torch


# def lr_adj(iter_num, change_num, lr_init, model, rate=0.2):
#     lr = lr_init
#     if iter_num < 400:
#         optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.01)
#     # if iter_num < 100:
#     #     optimizer = torch.optim.SGD(model.parameters(), lr=0.00025, momentum=0.9, weight_decay=0.1)
#     else:
#         if iter_num % change_num == 0 and iter_num != 0:
#             # if iter_num >= 600:
#             #     lr = lr * rate * 0.5
#             # else:
#             lr = lr*rate
#         else:
#             lr = lr
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
#
#     return optimizer, lr


def lr_adj(iter_num, change_num, lr, optimizer, rate=0.2, lr_warmup=0.00001, len_warmup=800):
    # step decay after 400 iterations
    if iter_num > len_warmup:
        if (iter_num-len_warmup) % change_num == 0:
            lr *= rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_warmup

    return optimizer, lr


