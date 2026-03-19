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

rate_train_H13  = [1251, 1242,  1244,  1252]
rate_train_H18  = [4000, 4561,  4000,  4000]
rate_train_AG   = [4000,        4000,  4000]
rate_train_TZ   = [4000,        4000,  4000]
rate_train_CHI  = [4000, 7997,  4000,   801]

rate_test = 0.999999999
slice_size = 3
class_num = 4

# readHSI --> normHSI_all --> label_trans --> data_split --> set_division --> spec_split --> data_trans

# H13+H18+AG+TZ--PU/PC
# Houston2013
path = './data/raw/City/Houston2013/'
image_name = 'Houston'
label_name = 'Houston_gt'
image_H13, label_H13 = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 先取430-860nm段，分辨率2nm，再插值到215
image_H13_new = image_H13[4:106]
image_H13_new = interpolate(image_H13_new, 215)

H13_label = label_trans(label_H13, [1, 5, 4, 9], [1, 2, 3, 4])
del label_H13
gc.collect()

H13_image_slice, H13_label_slice, __ = data_split(slice_size, image_H13_new, H13_label)
del image_H13_new, H13_label
gc.collect()

H13_img_shuffle, H13_gt_shuffle, __ = set_shuffle(H13_image_slice, H13_label_slice)
del H13_image_slice, H13_label_slice
gc.collect()

H13_image, H13_gt = set_division_pro(4, [1, 2, 3, 4], H13_img_shuffle, H13_gt_shuffle, 'train', rate_train_H13)
H13_image = normHSI_smp_s(H13_image)
del H13_img_shuffle, H13_gt_shuffle
gc.collect()

H13_gt_OH = one_hot_slice_domain(H13_gt, class_num=class_num, flag=1)
del H13_gt
gc.collect()

np.save(f"data/City_H1318/gen_H13/gt.npy", H13_gt_OH)
del H13_gt_OH
gc.collect()

np.save(f"data/City_H1318/gen_H13/img.npy", H13_image*2.8-0.45)

H13_SE_image = data_trans(H13_image*2.8-0.45)
del H13_image
gc.collect()

np.save(f"data/City_H1318/gen_H13/SE_img.npy", H13_SE_image)
del H13_SE_image
gc.collect()


# Houston2018
path = './data/raw/City/Houston2018/'
image_name = 'HoustonU'
label_name = 'HoustonU_gt'
image_H18_raw, label_H18 = readHSI(path, image_name, label_name, mode=1, img_order=[0, 1, 2])
image_H18 = interpolate(image_H18_raw[:48], 144)
image_H18_new = image_H18[4:106]
image_H18_new = interpolate(image_H18_new, 215)
del image_H18_raw, image_H18
gc.collect()

H18_label = label_trans(label_H18, [1, 6, 4, 10], [1, 2, 3, 4])
del label_H18
gc.collect()

H18_image_slice, H18_label_slice, __ = data_split(slice_size, image_H18_new, H18_label)
del image_H18_new, H18_label
gc.collect()

H18_img_shuffle, H18_gt_shuffle, __ = set_shuffle(H18_image_slice, H18_label_slice)
del H18_image_slice, H18_label_slice
gc.collect()

H18_image, H18_gt = set_division_pro(4, [1, 2, 3, 4], H18_img_shuffle, H18_gt_shuffle, 'train', rate_train_H18)
H18_image = normHSI_smp_s(H18_image)
del H18_img_shuffle, H18_gt_shuffle
gc.collect()

H18_gt_OH = one_hot_slice_domain(H18_gt, class_num=class_num, flag=2)
del H18_gt
gc.collect()

np.save(f"data/City_H1318/gen_H18/gt.npy", H18_gt_OH)
del H18_gt_OH
gc.collect()

np.save(f"data/City_H1318/gen_H18/img.npy", H18_image*2.2-0.05)

H18_SE_image = data_trans(H18_image*2.2-0.05)
del H18_image
gc.collect()

np.save(f"data/City_H1318/gen_H18/SE_img.npy", H18_SE_image)
del H18_SE_image
gc.collect()

# AG
path = './data/raw/City/AHU-Anguang/'
image_name = 'AG'
label_name = 'AG_gt'
image_AG, label_AG = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 先取430-860nm段，分辨率2nm，再插值到215
image_AG_new = image_AG[5:78]
image_AG_new = interpolate(image_AG_new, 215)
del image_AG
gc.collect()

AG_label = label_trans(label_AG, [1, 2, 3], [1, 3, 4])
del label_AG
gc.collect()

AG_image_slice, AG_label_slice, __ = data_split(slice_size, image_AG_new, AG_label)
del image_AG_new, AG_label
gc.collect()

AG_img_shuffle, AG_gt_shuffle, __ = set_shuffle(AG_image_slice, AG_label_slice)
del AG_image_slice, AG_label_slice
gc.collect()

AG_image, AG_gt = set_division_pro(3, [1, 3, 4], AG_img_shuffle, AG_gt_shuffle, 'train', rate_train_AG)
AG_image = normHSI_smp_s(AG_image)
del AG_img_shuffle, AG_gt_shuffle
gc.collect()

AG_gt_OH = one_hot_slice_domain(AG_gt, class_num=class_num, flag=3)
del AG_gt
gc.collect()

np.save(f"data/City_H1318/gen_AG/gt.npy", AG_gt_OH)
del AG_gt_OH
gc.collect()

np.save(f"data/City_H1318/gen_AG/img.npy", AG_image)

AG_SE_image = data_trans(AG_image)
del AG_image
gc.collect()

np.save(f"data/City_H1318/gen_AG/SE_img.npy", AG_SE_image)
del AG_SE_image
gc.collect()

# TZ
path = './data/raw/City/AHU-Tongzhou/'
image_name = 'tongzhou'
label_name = 'tongzhou_gt'
image_TZ, label_TZ = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 先取430-860nm段，分辨率2nm，再插值到215
image_TZ_new = image_TZ[5:78]
image_TZ_new = interpolate(image_TZ_new, 215)
del image_TZ
gc.collect()

TZ_label = label_trans(label_TZ, [1, 2, 3], [1, 3, 4])
del label_TZ
gc.collect()

TZ_image_slice, TZ_label_slice, __ = data_split(slice_size, image_TZ_new, TZ_label)
del image_TZ_new, TZ_label
gc.collect()

TZ_img_shuffle, TZ_gt_shuffle, __ = set_shuffle(TZ_image_slice, TZ_label_slice)
del TZ_image_slice, TZ_label_slice
gc.collect()

TZ_image, TZ_gt = set_division_pro(3, [1, 3, 4], TZ_img_shuffle, TZ_gt_shuffle, 'train', rate_train_TZ)
TZ_image = normHSI_smp_s(TZ_image)
del TZ_img_shuffle, TZ_gt_shuffle
gc.collect()

TZ_gt_OH = one_hot_slice_domain(TZ_gt, class_num=class_num, flag=4)
del TZ_gt
gc.collect()

np.save(f"data/City_H1318/gen_TZ/gt.npy", TZ_gt_OH)
del TZ_gt_OH
gc.collect()

np.save(f"data/City_H1318/gen_TZ/img.npy", TZ_image)

TZ_SE_image = data_trans(TZ_image)
del TZ_image
gc.collect()

np.save(f"data/City_H1318/gen_TZ/SE_img.npy", TZ_SE_image)
del TZ_SE_image
gc.collect()

# Chikusei
path = './data/raw/City/Chikusei/'
image_name = 'Chikusei'
label_name = 'Chikusei_gt'
image_CHI, label_CHI = readHSI(path, image_name, label_name, mode=1, img_order=[0, 1, 2])
# 先取430-860nm段，分辨率2nm，再插值到215
image_CHI_new = interpolate(image_CHI[13:97], 215)
del image_CHI
gc.collect()

CHI_label = label_trans(label_CHI, [8, 2, 3, 4, 7, 18], [1, 2, 2, 2, 3, 4])
del label_CHI
gc.collect()

CHI_image_slice, CHI_label_slice, __ = data_split(slice_size, image_CHI_new, CHI_label)
del image_CHI_new, CHI_label
gc.collect()

CHI_img_shuffle, CHI_gt_shuffle, __ = set_shuffle(CHI_image_slice, CHI_label_slice)
del CHI_image_slice, CHI_label_slice
gc.collect()

CHI_image, CHI_gt = set_division_pro(4, [1, 2, 3, 4], CHI_img_shuffle, CHI_gt_shuffle, 'train', rate_train_CHI)
CHI_image = normHSI_smp_s(CHI_image)
del CHI_img_shuffle, CHI_gt_shuffle
gc.collect()

CHI_gt_OH = one_hot_slice_domain(CHI_gt, class_num=class_num, flag=5)
del CHI_gt
gc.collect()

np.save(f"data/City_H1318/gen_CHI/gt.npy", CHI_gt_OH)
del CHI_gt_OH
gc.collect()

np.save(f"data/City_H1318/gen_CHI/img.npy", CHI_image)

CHI_SE_image = data_trans(CHI_image)
del CHI_image
gc.collect()

np.save(f"data/City_H1318/gen_CHI/SE_img.npy", CHI_SE_image)
del CHI_SE_image
gc.collect()

path = './data/raw/City/PaviaU/'
image_name = 'paviaU'
label_name = 'paviaU_gt'
image_PU, label_PU = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 先取430-860nm段，分辨率2nm，再插值到215
image_PU_new = interpolate(image_PU, 215)
del image_PU
gc.collect()

PU_label = label_trans(label_PU, [2, 6, 4, 1], [1, 2, 3, 4])
del label_PU
gc.collect()

PU_image_slice, PU_label_slice, PU_row_col = data_split(slice_size, image_PU_new, PU_label)
del image_PU_new, PU_label
gc.collect()

PU_image, PU_gt, PU_point_idx = set_division(4, [1, 2, 3, 4], PU_image_slice, PU_label_slice, 'train', rate_test, PU_row_col)
PU_image = normHSI_smp_s(PU_image)
del PU_image_slice, PU_label_slice
gc.collect()


PU_gt_OH = one_hot_slice(PU_gt, class_num=class_num)
del PU_gt
gc.collect()

np.save(f"data/City_H1318/gen_PU/gt.npy", torch.cat((PU_gt_OH, PU_point_idx), dim=1))
del PU_gt_OH
gc.collect()

np.save(f"data/City_H1318/gen_PU/img.npy", PU_image*1.5-0.1)

PU_SE_image = data_trans(PU_image*1.5-0.1)
del PU_image
gc.collect()

np.save(f"data/City_H1318/gen_PU/SE_img.npy", PU_SE_image)
del PU_SE_image
gc.collect()


# PC
path = './data/raw/City/PaviaC/'
image_name = 'pavia'
label_name = 'pavia_gt'
image_PC, label_PC = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 先取430-860nm段，分辨率2nm，再插值到215
image_PC_new = interpolate(image_PC, 215)
del image_PC
gc.collect()

PC_label = label_trans(label_PC, [3, 5, 2, 6], [1, 2, 3, 4])
del label_PC
gc.collect()

PC_image_slice, PC_label_slice, PC_row_col = data_split(slice_size, image_PC_new, PC_label)
del image_PC_new, PC_label
gc.collect()

PC_image, PC_gt, PC_point_idx = set_division(4, [1, 2, 3, 4], PC_image_slice, PC_label_slice, 'train', rate_test, PC_row_col)
PC_image = normHSI_smp_s(PC_image)
del PC_image_slice, PC_label_slice
gc.collect()

PC_gt_OH = one_hot_slice(PC_gt, class_num=class_num)
del PC_gt
gc.collect()

np.save(f"data/City_H1318/gen_PC/gt.npy", torch.cat((PC_gt_OH, PC_point_idx), dim=1))
del PC_gt_OH
gc.collect()

np.save(f"data/City_H1318/gen_PC/img.npy", PC_image* 1.5 - 0.1)

PC_SE_image = data_trans(PC_image* 1.5 - 0.1)
del PC_image
gc.collect()

np.save(f"data/City_H1318/gen_PC/SE_img.npy", PC_SE_image)
del PC_SE_image
gc.collect()
#




