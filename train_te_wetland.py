import os
import gc
import time
import random
import warnings
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from networks.SMDGnet import SMDGnet
from torch.utils.data import DataLoader
from loss import simi_cal, DynamicSDPLoss, AlignLoss
from ImageDataset import build_loader
from utils.lr_adjust import lr_adj
from sklearn.metrics import accuracy_score
from utils.draw_loss_curve import draw_loss_curve
from utils.ema import EMA


GF5YC_SE_img = np.load("data/Wetland/gen_GF5YC/SE_img.npy")
GF5YC_SE_img = torch.from_numpy(GF5YC_SE_img).float()
GF5YC_img = np.load("data/Wetland/gen_GF5YC/img.npy")
GF5YC_img = torch.from_numpy(GF5YC_img).float()
GF5YC_label = np.load("data/Wetland/gen_GF5YC/gt.npy")
GF5YC_label = torch.LongTensor(GF5YC_label)

HHK20_SE_img = np.load("data/Wetland/gen_HHK20/SE_img.npy")
HHK20_SE_img = torch.from_numpy(HHK20_SE_img).float()
HHK20_img = np.load("data/Wetland/gen_HHK20/img.npy")
HHK20_img = torch.from_numpy(HHK20_img).float()
HHK20_label = np.load("data/Wetland/gen_HHK20/gt.npy")
HHK20_label = torch.LongTensor(HHK20_label)

HHK21_SE_img = np.load("data/Wetland/gen_HHK21/SE_img.npy")
HHK21_SE_img = torch.from_numpy(HHK21_SE_img).float()
HHK21_img = np.load("data/Wetland/gen_HHK21/img.npy")
HHK21_img = torch.from_numpy(HHK21_img).float()
HHK21_label = np.load("data/Wetland/gen_HHK21/gt.npy")
HHK21_label = torch.LongTensor(HHK21_label)

Loukia_SE_img = np.load("data/Wetland/gen_Loukia/SE_img.npy")
Loukia_SE_img = torch.from_numpy(Loukia_SE_img).float()
Loukia_img = np.load("data/Wetland/gen_Loukia/img.npy")
Loukia_img = torch.from_numpy(Loukia_img).float()
Loukia_label = np.load("data/Wetland/gen_Loukia/gt.npy")
Loukia_label = torch.LongTensor(Loukia_label)

Dioni_SE_img = np.load("data/Wetland/gen_Dioni/SE_img.npy")
Dioni_SE_img = torch.from_numpy(Dioni_SE_img).float()
Dioni_img = np.load("data/Wetland/gen_Dioni/img.npy")
Dioni_img = torch.from_numpy(Dioni_img).float()
Dioni_label = np.load("data/Wetland/gen_Dioni/gt.npy")
Dioni_label = torch.LongTensor(Dioni_label)

SE_img = [GF5YC_SE_img, HHK20_SE_img, HHK21_SE_img, Loukia_SE_img, Dioni_SE_img]
img = [GF5YC_img, HHK20_img, HHK21_img, Loukia_img, Dioni_img]
label = [GF5YC_label, HHK20_label, HHK21_label, Loukia_label, Dioni_label]

del GF5YC_SE_img, HHK20_SE_img, HHK21_SE_img, Loukia_SE_img, Dioni_SE_img, GF5YC_img, HHK20_img, HHK21_img,\
    Loukia_img, Dioni_img, GF5YC_label, HHK20_label, HHK21_label, Loukia_label, Dioni_label
gc.collect()


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

warnings.filterwarnings("ignore")

w = [3, 6, 1, 1]
num_epoch = 5
batch_size = 1024
learning_rate = 0.002
rate = 0.5
change_num = 250
ema_epoch = 3
device = "cuda:0"

slice_size = 3
in_ch = slice_size**2
out_ch_shift = 256
out_ch_ifem = [64, 64, 128, 128]
out_ch = [256, 64]
img_size = 70
band_num = 210
class_num = 7
domains_per_batch = 5
classes_per_domain = 7
samples_per_class = 29
num_batches_per_epoch = 400

work_dir = f'./work_dir/Wetland_inch{in_ch}_w{w[0]}-{w[1]}-{w[2]}-{w[3]}_b{batch_size}_lr{learning_rate}/' \
           + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 记录损失
logs_path = work_dir + 'logs/'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
with open(logs_path + "train_logs.txt", 'a') as f:
    f.write("Training loss logs:")
    f.write("\n")
f.close()

train_loss_list = []

train_loader = build_loader(SE_img, img, label, batch_size, domains_per_batch=domains_per_batch,
                            classes_per_domain=classes_per_domain, samples_per_class=samples_per_class,
                            num_batches_per_epoch=num_batches_per_epoch)

model = SMDGnet(in_ch=in_ch, out_ch_shift=out_ch_shift, out_ch_ifem=out_ch_ifem, out_ch=out_ch,
                img_size=img_size, band_num=band_num, class_num=class_num, slice_size=slice_size).to(device)

loss_classify = nn.BCEWithLogitsLoss(reduction='mean')
loss_diff = nn.BCEWithLogitsLoss(reduction='mean')
loss_prog = DynamicSDPLoss()
loss_align = AlignLoss(kernel_mul=2.0, kernel_num=5, fix_sigma=None)

# start training
time_start = time.time()

for epoch in range(0, num_epoch):
    if epoch == ema_epoch:
        ema = EMA(model, decay=0.9)
        ema.register()
    f = open(logs_path + "train_logs.txt", 'a')
    epoch_start_time = time.time()
    train_loss = 0.0

    model.train()

    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    loop_len = len(loop)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.1, nesterov=True)

    for i, data in loop:
        num_iters = epoch * loop_len + i
        optimizer, learning_rate = lr_adj(num_iters, change_num, learning_rate, optimizer, rate, lr_warmup=0.000001,
                                          len_warmup=0)
        optimizer.zero_grad()

        y_1, side_1_1, side_1_2, side_1_3, side_1_4, y_shift_1 \
            = model(data[0].to(device), data[1].to(device))
        y_2, side_2_1, side_2_2, side_2_3, side_2_4, y_shift_2 \
            = model(data[3].to(device), data[4].to(device))

        simi_gt = torch.full([data[2].shape[0], data[2].shape[0]], 0.).to(device)
        simi_SDP_gt = torch.full([data[2].shape[0], data[2].shape[0]], -1.).to(device)
        for batch in range(data[2].shape[0]):
            cons = torch.where(torch.argmax(data[5][:, :-1], dim=1) == torch.argmax(data[2][:, :-1][batch]))
            simi_gt[batch, cons[0]] = 1
            simi_SDP_gt[batch, cons[0]] = 1

        # loss 1
        loss1_1 = loss_classify(y_1, data[2][:, :-1].float().to(device))
        loss1_2 = loss_classify(y_2, data[5][:, :-1].float().to(device))

        loss1 = loss1_1 + loss1_2

        simi_1 = simi_cal(side_1_1, side_2_1)
        simi_2 = simi_cal(side_1_2, side_2_2)
        simi_3 = simi_cal(side_1_3, side_2_3)
        simi_4 = simi_cal(side_1_4, side_2_4)

        similarity_all = [simi_1, simi_2, simi_3, simi_4]

        # loss 2
        loss2_1 = loss_diff(simi_1, simi_gt)
        loss2_2 = loss_diff(simi_2, simi_gt)
        loss2_3 = loss_diff(simi_3, simi_gt)
        loss2_4 = loss_diff(simi_4, simi_gt)

        loss2 = loss2_1 + loss2_2 + loss2_3 + loss2_4

        # loss 3
        loss3 = loss_prog(similarity_all, simi_SDP_gt)

        # loss 4
        loss4 = loss_align(y_shift_1, y_shift_2)

        batch_loss = w[0] * loss1 + w[1] * loss2 + w[2] * loss3 + w[3] * loss4

        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss = train_loss + batch_loss.item()
            gt_1 = data[2][:, :-1].argmax(dim=1).flatten().cpu().numpy()
            pred_1 = y_1.argmax(dim=1).flatten().cpu().numpy()
            gt_2 = data[5][:, :-1].argmax(dim=1).flatten().cpu().numpy()
            pred_2 = y_2.argmax(dim=1).flatten().cpu().numpy()
            oa_1 = accuracy_score(gt_1, pred_1)
            oa_2 = accuracy_score(gt_2, pred_2)

        # 设置tqdm显示
        loop.set_description(f'Epoch [{epoch + 1}/{num_epoch}]')
        loop.set_postfix(cls_ls=loss1.item(), dif_ls=loss2.item(), prog_ls=loss3.item(), ang_ls=loss4.item(),
                         b_ls=batch_loss.item(), lr=optimizer.state_dict()['param_groups'][0]['lr'],
                         oa_1=oa_1, oa_2=oa_2)
        optimizer.zero_grad()

    if epoch >= ema_epoch:
        ema.apply_shadow()
    models_path = work_dir + 'models/'
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    torch.save(model.state_dict(), models_path + 'model{}.pth'.format(epoch + 1))

    print('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f' %
          (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss))

    f.write('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f \n' %
            (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss))
    f.close()

    # training loss
    train_loss_list.append(train_loss)

    if epoch >= ema_epoch:
        ema.restore()

time_end = time.time()
f.close()

# draw the loss curve
epoch_list = [(i + 1) for i in range(num_epoch)]
draw_loss_curve(epoch_list, train_loss=train_loss_list, save_path=logs_path + "loss.png")
print("training time:", time_end - time_start, 's')
f.close()
