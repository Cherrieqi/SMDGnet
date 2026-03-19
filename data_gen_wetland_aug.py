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
from utils.data_aug import data_aug


rate_train_GF5YC   = [0.99999999, 0.99999999, 0.99999999, 0.99999999,  600, 0.99999999]
# rate_train_GF5YC = [       360,        217,        132,        832,  600,   500]
GF5YC_aug_num      = [      1000,        600,        800,        700,         500]

rate_train_HHK20   = [0.99999999, 0.99999999, 0.99999999, 0.99999999,  600,             0.99999999]
# rate_train_HHK20 = [       398,        584,        310,        508,  600,             853]
HHK20_aug_num      = [      1000,        400,        600,       1000,                   450]

rate_train_HHK21   = [            0.99999999, 0.99999999,              600,             0.99999999]
# rate_train_HHK21 = [                   941,        400,              600,             1184]
HHK21_aug_num      = [                   200,        800,                               450]

rate_train_Loukia  = [                                                 600, 0.99999999]
# rate_train_Loukia= [                                                 600,   451]
Loukia_aug_num     = [                                                        400]

rate_train_Dioni   = [                                                 600, 0.99999999]
# rate_train_Dioni = [                                                 600,   398]
Dioni_aug_num      = [                                                        400]

rate_test = 0.999999999
slice_size = 3
class_num = 7

# readHSI --> normHSI_all --> label_trans --> data_split --> set_division --> spec_split --> data_trans

# GF5YC+HHK20+HHK21+Loukia+Dioni---ZY1YC
# GF5YC
path = './data/raw/Wetland/GF5Yancheng/'
image_name = 'GF_YC_data'
label_name = 'GF_YC_gt'
image_GF5YC, label_GF5YC = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 400-2500nm段，分辨率10nm，插值到210
image_GF5YC_new = interpolate(image_GF5YC, 210)
del image_GF5YC
gc.collect()

GF5YC_label = label_trans(label_GF5YC, [1, 2, 3, 4, 6, 7], [1, 2, 3, 4, 5, 6])
del label_GF5YC
gc.collect()

GF5YC_image_slice, GF5YC_label_slice, __ = data_split(slice_size, image_GF5YC_new, GF5YC_label)
del image_GF5YC_new, GF5YC_label
gc.collect()

GF5YC_img_shuffle, GF5YC_gt_shuffle, __ = set_shuffle(GF5YC_image_slice, GF5YC_label_slice)
del GF5YC_image_slice, GF5YC_label_slice
gc.collect()

GF5YC_image, GF5YC_gt = set_division_pro(6, [1, 2, 3, 4, 5, 6], GF5YC_img_shuffle, GF5YC_gt_shuffle, 'train', rate_train_GF5YC)
del GF5YC_img_shuffle, GF5YC_gt_shuffle
gc.collect()

GF5YC_image_aug, GF5YC_gt_aug = data_aug(GF5YC_image, GF5YC_gt, class_list=[1,2,3,4,6], num=GF5YC_aug_num)
GF5YC_image = torch.cat((GF5YC_image, GF5YC_image_aug), dim=0)
GF5YC_image = normHSI_smp_s(GF5YC_image)
GF5YC_gt = torch.cat((GF5YC_gt, GF5YC_gt_aug), dim=0)
del GF5YC_image_aug, GF5YC_gt_aug
gc.collect()

GF5YC_gt_OH = one_hot_slice_domain(GF5YC_gt, class_num=class_num, flag=1)
del GF5YC_gt
gc.collect()

np.save(f"data/Wetland/gen_GF5YC/gt.npy", GF5YC_gt_OH)
del GF5YC_gt_OH
gc.collect()

np.save(f"data/Wetland/gen_GF5YC/img.npy", GF5YC_image*2-0.05)

GF5YC_SE_image = data_trans(GF5YC_image*2-0.05)
del GF5YC_image
gc.collect()

np.save(f"data/Wetland/gen_GF5YC/SE_img.npy", GF5YC_SE_image)
del GF5YC_SE_image
gc.collect()


# HHK2020
path = './data/raw/Wetland/HHK2020/'
image_name = 'ZY_HHK_data108_20200628'
label_name = 'ZY_HHK_gt108_20200628'
label_name_1 = 'ZY_HHK_gt6'
image_HHK20, label_HHK20 = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
__, label_HHK20_1 = readHSI(path, image_name, label_name_1, mode=0, img_order=[2, 0, 1])
# 400-2500nm段，分辨率10nm，插值到210
image_HHK20_new = interpolate(image_HHK20, 210)
del image_HHK20
gc.collect()

HHK20_label = label_trans(label_HHK20, [6, 1, 7, 2, 3, 4], [2, 3, 5, 7, 7, 7])
HHK20_label_1 = label_trans(label_HHK20_1, [1, 2], [1, 4])
del label_HHK20, label_HHK20_1
gc.collect()

HHK20_image_slice, HHK20_label_slice, __ = data_split(slice_size, image_HHK20_new, HHK20_label+HHK20_label_1)
del image_HHK20_new, HHK20_label
gc.collect()

HHK20_img_shuffle, HHK20_gt_shuffle, __ = set_shuffle(HHK20_image_slice, HHK20_label_slice)
del HHK20_image_slice, HHK20_label_slice
gc.collect()

HHK20_image, HHK20_gt = set_division_pro(6, [1, 2, 3, 4, 5, 7], HHK20_img_shuffle, HHK20_gt_shuffle, 'train', rate_train_HHK20)
del HHK20_img_shuffle, HHK20_gt_shuffle
gc.collect()

HHK20_image_aug, HHK20_gt_aug = data_aug(HHK20_image, HHK20_gt, class_list=[1,2,3,4,7], num=HHK20_aug_num)
HHK20_image = torch.cat((HHK20_image, HHK20_image_aug), dim=0)
HHK20_image = normHSI_smp_s(HHK20_image)
HHK20_gt = torch.cat((HHK20_gt, HHK20_gt_aug), dim=0)
del HHK20_image_aug, HHK20_gt_aug
gc.collect()

HHK20_gt_OH = one_hot_slice_domain(HHK20_gt, class_num=class_num, flag=2)
del HHK20_gt
gc.collect()

np.save(f"data/Wetland/gen_HHK20/gt.npy", HHK20_gt_OH)
del HHK20_gt_OH
gc.collect()

np.save(f"data/Wetland/gen_HHK20/img.npy", HHK20_image)

HHK20_SE_image = data_trans(HHK20_image)
del HHK20_image
gc.collect()

np.save(f"data/Wetland/gen_HHK20/SE_img.npy", HHK20_SE_image)
del HHK20_SE_image
gc.collect()


# HHK2021
path = './data/raw/Wetland/HHK2021/'
image_name = 'ZY_HHK_data108_20210929'
label_name = 'ZY_HHK_gt108_20210929'
image_HHK21, label_HHK21 = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 400-2500nm段，分辨率10nm，插值到210
image_HHK21_new = interpolate(image_HHK21, 210)
del image_HHK21
gc.collect()

HHK21_label = label_trans(label_HHK21, [6, 1, 7, 2, 3, 4], [2, 3, 5, 7, 7, 7])
del label_HHK21
gc.collect()

HHK21_image_slice, HHK21_label_slice, __ = data_split(slice_size, image_HHK21_new, HHK21_label)
del image_HHK21_new, HHK21_label
gc.collect()

HHK21_img_shuffle, HHK21_gt_shuffle, __ = set_shuffle(HHK21_image_slice, HHK21_label_slice)
del HHK21_image_slice, HHK21_label_slice
gc.collect()

HHK21_image, HHK21_gt = set_division_pro(4, [2, 3, 5, 7], HHK21_img_shuffle, HHK21_gt_shuffle, 'train', rate_train_HHK21)
del HHK21_img_shuffle, HHK21_gt_shuffle
gc.collect()

HHK21_image_aug, HHK21_gt_aug = data_aug(HHK21_image, HHK21_gt, class_list=[2,3,7], num=HHK21_aug_num)
HHK21_image = torch.cat((HHK21_image, HHK21_image_aug), dim=0)
HHK21_image = normHSI_smp_s(HHK21_image)
HHK21_gt = torch.cat((HHK21_gt, HHK21_gt_aug), dim=0)
del HHK21_image_aug, HHK21_gt_aug
gc.collect()

HHK21_gt_OH = one_hot_slice_domain(HHK21_gt, class_num=class_num, flag=3)
del HHK21_gt
gc.collect()

np.save(f"data/Wetland/gen_HHK21/gt.npy", HHK21_gt_OH)
del HHK21_gt_OH
gc.collect()

np.save(f"data/Wetland/gen_HHK21/img.npy", HHK21_image)

HHK21_SE_image = data_trans(HHK21_image)
del HHK21_image
gc.collect()

np.save(f"data/Wetland/gen_HHK21/SE_img.npy", HHK21_SE_image)
del HHK21_SE_image
gc.collect()


# Loukia
path = './data/raw/Wetland/Loukia/'
image_name = 'Loukia'
label_name = 'Loukia_gt'
image_Loukia, label_Loukia = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 400-2500nm段，分辨率10nm，插值到210
image_Loukia_new = interpolate(image_Loukia, 210)
del image_Loukia
gc.collect()

Loukia_label = label_trans(label_Loukia, [13, 14], [5, 6])
del label_Loukia
gc.collect()

Loukia_image_slice, Loukia_label_slice, __ = data_split(slice_size, image_Loukia_new, Loukia_label)
del image_Loukia_new, Loukia_label
gc.collect()

Loukia_img_shuffle, Loukia_gt_shuffle, __ = set_shuffle(Loukia_image_slice, Loukia_label_slice)
del Loukia_image_slice, Loukia_label_slice
gc.collect()

Loukia_image, Loukia_gt = set_division_pro(2, [5, 6], Loukia_img_shuffle, Loukia_gt_shuffle, 'train', rate_train_Loukia)
del Loukia_img_shuffle, Loukia_gt_shuffle
gc.collect()

Loukia_image_aug, Loukia_gt_aug = data_aug(Loukia_image, Loukia_gt, class_list=[6], num=Loukia_aug_num)
Loukia_image = torch.cat((Loukia_image, Loukia_image_aug), dim=0)
Loukia_image = normHSI_smp_s(Loukia_image)
Loukia_gt = torch.cat((Loukia_gt, Loukia_gt_aug), dim=0)
del Loukia_image_aug, Loukia_gt_aug
gc.collect()

Loukia_gt_OH = one_hot_slice_domain(Loukia_gt, class_num=class_num, flag=4)
del Loukia_gt
gc.collect()

np.save(f"data/Wetland/gen_Loukia/gt.npy", Loukia_gt_OH)
del Loukia_gt_OH
gc.collect()

np.save(f"data/Wetland/gen_Loukia/img.npy", Loukia_image)

Loukia_SE_image = data_trans(Loukia_image)
del Loukia_image
gc.collect()

np.save(f"data/Wetland/gen_Loukia/SE_img.npy", Loukia_SE_image)
del Loukia_SE_image
gc.collect()

# Dioni
path = './data/raw/Wetland/Dioni/'
image_name = 'Dioni'
label_name = 'Dioni_gt'
image_Dioni, label_Dioni = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
# 400-2500nm段，分辨率10nm，插值到210
image_Dioni_new = interpolate(image_Dioni, 210)
del image_Dioni
gc.collect()

Dioni_label = label_trans(label_Dioni, [13, 14], [5, 6])
del label_Dioni
gc.collect()

Dioni_image_slice, Dioni_label_slice, __ = data_split(slice_size, image_Dioni_new, Dioni_label)
del image_Dioni_new, Dioni_label
gc.collect()

Dioni_img_shuffle, Dioni_gt_shuffle, __ = set_shuffle(Dioni_image_slice, Dioni_label_slice)
del Dioni_image_slice, Dioni_label_slice
gc.collect()

Dioni_image, Dioni_gt = set_division_pro(2, [5, 6], Dioni_img_shuffle, Dioni_gt_shuffle, 'train', rate_train_Dioni)
del Dioni_img_shuffle, Dioni_gt_shuffle
gc.collect()

Dioni_image_aug, Dioni_gt_aug = data_aug(Dioni_image, Dioni_gt, class_list=[6], num=Dioni_aug_num)
Dioni_image = torch.cat((Dioni_image, Dioni_image_aug), dim=0)
Dioni_image = normHSI_smp_s(Dioni_image)
Dioni_gt = torch.cat((Dioni_gt, Dioni_gt_aug), dim=0)
del Dioni_image_aug, Dioni_gt_aug
gc.collect()

Dioni_gt_OH = one_hot_slice_domain(Dioni_gt, class_num=class_num, flag=5)
del Dioni_gt
gc.collect()

np.save(f"data/Wetland/gen_Dioni/gt.npy", Dioni_gt_OH)
del Dioni_gt_OH
gc.collect()

np.save(f"data/Wetland/gen_Dioni/img.npy", Dioni_image)

Dioni_SE_image = data_trans(Dioni_image)
del Dioni_image
gc.collect()

np.save(f"data/Wetland/gen_Dioni/SE_img.npy", Dioni_SE_image)
del Dioni_SE_image
gc.collect()


# ZY1YC
path = './data/raw/Wetland/ZY1Yancheng/'
image_name = 'ZY_YC_data147'
label_name = 'ZY_YC_gt147'
label_name_1 = 'ZY_YC_gt119'
image_ZY1YC, label_ZY1YC = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])
__, label_ZY1YC_1 = readHSI(path, image_name, label_name_1, mode=0, img_order=[2, 0, 1])
# 400-2500nm段，分辨率10nm，插值到210
image_ZY1YC_new = interpolate(image_ZY1YC, 210)
del image_ZY1YC
gc.collect()

ZY1YC_label = label_trans(label_ZY1YC, [1, 2, 3, 4, 6, 7], [1, 2, 3, 4, 5, 6])
ZY1YC_label_1 = label_trans(label_ZY1YC_1, [6], [7])
del label_ZY1YC, label_ZY1YC_1
gc.collect()

ZY1YC_image_slice, ZY1YC_label_slice, ZY1YC_row_col = data_split(slice_size, image_ZY1YC_new, ZY1YC_label+ZY1YC_label_1)
del image_ZY1YC_new, ZY1YC_label
gc.collect()

ZY1YC_image, ZY1YC_gt, ZY1YC_point_idx = set_division(7, [1, 2, 3, 4, 5, 6, 7], ZY1YC_image_slice, ZY1YC_label_slice, 'train', rate_test, ZY1YC_row_col)
ZY1YC_image = normHSI_smp_s(ZY1YC_image)
del ZY1YC_image_slice, ZY1YC_label_slice
gc.collect()

ZY1YC_gt_OH = one_hot_slice(ZY1YC_gt, class_num=class_num)
del ZY1YC_gt
gc.collect()

np.save(f"data/Wetland/gen_ZY1YC/gt.npy", torch.cat((ZY1YC_gt_OH, ZY1YC_point_idx), dim=1))
del ZY1YC_gt_OH
gc.collect()

np.save(f"data/Wetland/gen_ZY1YC/img.npy", ZY1YC_image)

ZY1YC_SE_image = data_trans(ZY1YC_image)
del ZY1YC_image
gc.collect()

np.save(f"data/Wetland/gen_ZY1YC/SE_img.npy", ZY1YC_SE_image)
del ZY1YC_SE_image
gc.collect()






