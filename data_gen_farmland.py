import gc
import torch
import numpy as np

from utils.data_split import data_split
from utils.data_trans import data_trans
from utils.readHSI import readHSI, label_trans, one_hot_slice_domain, one_hot_slice
from utils.normHSI import normHSI_all, normHSI_smp_s
from utils.interpolate import interpolate
from utils.data_shuffle import data_shuffle
from utils.set_slc_division import set_division, set_division_pro, set_shuffle


rate_train_HC  = [10287, 11800, 10000]
rate_train_IP  = [2455]
rate_train_CHI = [        801, 2845]

rate_test = 0.999999999
slice_size = 3
class_num = 3

# readHSI --> normHSI_all --> label_trans --> data_split --> set_division --> spec_split --> data_trans

# HC+IP+CHI--LK
# HC
path = './data/raw/Farmland/Hanchuan/'
image_name = 'WHU_Hi_HanChuan'
label_name = 'WHU_Hi_HanChuan_gt'
image_HC, label_HC = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 先取400-1000nm段，分辨率2nm，再插值到300
image_HC_new = interpolate(image_HC, 300)
del image_HC
gc.collect()

HC_label = label_trans(label_HC, [3, 14, 16], [1, 2, 3])
del label_HC
gc.collect()

HC_image_slice, HC_label_slice, __ = data_split(slice_size, image_HC_new, HC_label)
del image_HC_new, HC_label
gc.collect()

HC_img_shuffle, HC_gt_shuffle, __ = set_shuffle(HC_image_slice, HC_label_slice)
del HC_image_slice, HC_label_slice
gc.collect()

HC_image, HC_gt = set_division_pro(3, [1, 2, 3], HC_img_shuffle, HC_gt_shuffle, 'train', rate_train_HC)
HC_image = normHSI_smp_s(HC_image)
del HC_img_shuffle, HC_gt_shuffle
gc.collect()

HC_gt_OH = one_hot_slice_domain(HC_gt, class_num=class_num, flag=1)
del HC_gt
gc.collect()

np.save(f"data/Farmland/gen_HC/gt.npy", HC_gt_OH)
del HC_gt_OH
gc.collect()

np.save(f"data/Farmland/gen_HC/img.npy", HC_image)

HC_SE_image = data_trans(HC_image)
del HC_image
gc.collect()

np.save(f"data/Farmland/gen_HC/SE_img.npy", HC_SE_image)
del HC_SE_image
gc.collect()

# IP
path = './data/raw/Farmland/India_pines/'
image_name = 'Indian_pines_corrected'
label_name = 'Indian_pines_gt'
image_IP, label_IP = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 先取400-1000nm段，分辨率2nm，再插值到300
image_IP_new = interpolate(image_IP[:60], 300)
del image_IP
gc.collect()

IP_label = label_trans(label_IP, [11], [1])
del label_IP
gc.collect()

IP_image_slice, IP_label_slice, __ = data_split(slice_size, image_IP_new, IP_label)
del image_IP_new, IP_label
gc.collect()

IP_img_shuffle, IP_gt_shuffle, __ = set_shuffle(IP_image_slice, IP_label_slice)
del IP_image_slice, IP_label_slice
gc.collect()

IP_image, IP_gt = set_division_pro(1, [1], IP_img_shuffle, IP_gt_shuffle, 'train', rate_train_IP)
IP_image = normHSI_smp_s(IP_image)
del IP_img_shuffle, IP_gt_shuffle
gc.collect()

IP_gt_OH = one_hot_slice_domain(IP_gt, class_num=class_num, flag=2)
del IP_gt
gc.collect()

np.save(f"data/Farmland/gen_IP/gt.npy", IP_gt_OH)
del IP_gt_OH
gc.collect()

np.save(f"data/Farmland/gen_IP/img.npy", IP_image)

IP_SE_image = data_trans(IP_image)
del IP_image
gc.collect()

np.save(f"data/Farmland/gen_IP/SE_img.npy", IP_SE_image)
del IP_SE_image
gc.collect()

# Chikusei
path = './data/raw/Farmland/Chikusei/'
image_name = 'Chikusei'
label_name = 'Chikusei_gt'
image_CHI, label_CHI = readHSI(path, image_name, label_name, mode=1, img_order=[0, 1, 2])
# 先取400-1000nm段，分辨率2nm，再插值到300
image_CHI_new = interpolate(image_CHI[4:124], 300)
del image_CHI
gc.collect()

CHI_label = label_trans(label_CHI, [18, 1], [2, 3])
del label_CHI
gc.collect()

CHI_image_slice, CHI_label_slice, __ = data_split(slice_size, image_CHI_new, CHI_label)
del image_CHI_new, CHI_label
gc.collect()

CHI_img_shuffle, CHI_gt_shuffle, __ = set_shuffle(CHI_image_slice, CHI_label_slice)
del CHI_image_slice, CHI_label_slice
gc.collect()

CHI_image, CHI_gt = set_division_pro(2, [2, 3], CHI_img_shuffle, CHI_gt_shuffle, 'train', rate_train_CHI)
CHI_image = normHSI_smp_s(CHI_image)
del CHI_img_shuffle, CHI_gt_shuffle
gc.collect()

CHI_gt_OH = one_hot_slice_domain(CHI_gt, class_num=class_num, flag=3)
del CHI_gt
gc.collect()

np.save(f"data/Farmland/gen_CHI/gt.npy", CHI_gt_OH)
del CHI_gt_OH
gc.collect()

np.save(f"data/Farmland/gen_CHI/img.npy", CHI_image)

CHI_SE_image = data_trans(CHI_image)
del CHI_image
gc.collect()

np.save(f"data/Farmland/gen_CHI/SE_img.npy", CHI_SE_image)
del CHI_SE_image
gc.collect()


# Longkou
path = './data/raw/Farmland/Longkou/'
image_name = 'WHU_Hi_LongKou'
label_name = 'WHU_Hi_LongKou_gt'
image_LK, label_LK = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 先取400-1000nm段，分辨率2nm，再插值到300
image_LK_new = interpolate(image_LK, 300)

LK_label = label_trans(label_LK, [4, 8, 7], [1, 2, 3])
del label_LK
gc.collect()

LK_image_slice, LK_label_slice, LK_row_col = data_split(slice_size, image_LK_new, LK_label)
del image_LK_new, LK_label
gc.collect()

LK_image, LK_gt, LK_point_idx = set_division(3, [1, 2, 3], LK_image_slice, LK_label_slice, 'train', rate_test, LK_row_col)
LK_image = normHSI_smp_s(LK_image)
del LK_image_slice, LK_label_slice
gc.collect()

LK_gt_OH = one_hot_slice(LK_gt, class_num=class_num)
del LK_gt
gc.collect()

np.save(f"data/Farmland/gen_LK/gt.npy", torch.cat((LK_gt_OH, LK_point_idx), dim=1))
del LK_gt_OH
gc.collect()

np.save(f"data/Farmland/gen_LK/img.npy", LK_image*1.3-0.08)

LK_SE_image = data_trans(LK_image*1.3-0.08)
del LK_image
gc.collect()

np.save(f"data/Farmland/gen_LK/SE_img.npy", LK_SE_image)
del LK_SE_image
gc.collect()
