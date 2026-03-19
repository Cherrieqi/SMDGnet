import torch


def data_trans(ori_img_slice):
    """
    Spatial-spectral exchange
    :param ori_img_slice: tensor, input HSI img slice [N, c, slice_size, slice_size]
    :return: trans_img_slice: tensor, output HSI img slice [N, slice_size*slice_size, (c+2)/3, (c+2)/3]
    """
    N, c, slice_size, __ = ori_img_slice.shape
    slice_size_new = (c+2)//3
    trans_img_slice = torch.zeros(N, slice_size ** 2, slice_size_new, slice_size_new)
    ori_img_slice = ori_img_slice.reshape(N, c, -1)
    ori_img_slice = ori_img_slice.permute(0, 2, 1)

    for cc in range(slice_size_new):
        trans_img_slice[:, :, cc] = ori_img_slice[:, :, cc:cc + slice_size_new*2:2]

    return trans_img_slice


# def data_trans(ori_img_slice):
#     """
#     Spatial-spectral exchange
#     :param ori_img_slice: tensor, input HSI img slice [N, c, slice_size, slice_size]
#     :return: trans_img_slice: tensor, output HSI img slice [N, slice_size*slice_size, (c+2)/3, (c+2)/3]
#     """
#     c, slice_size, __ = ori_img_slice.shape
#     slice_size_new = (c+2)//3
#     trans_img_slice = torch.zeros(slice_size ** 2, slice_size_new, slice_size_new)
#     ori_img_slice = ori_img_slice.reshape(c, -1)
#     ori_img_slice = ori_img_slice.permute(1, 0)
#
#     for cc in range(slice_size_new):
#         trans_img_slice[:, cc] = ori_img_slice[:, cc:cc + slice_size_new*2:2]
#
#     return trans_img_slice


if __name__ == '__main__':
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7]).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    # x = x.repeat(2, 1, 1, 1)
    x = x.repeat(2, 1, 2, 2)
    x[1] = (torch.tensor([8, 9, 10, 11, 12, 13, 14]).unsqueeze(1).unsqueeze(1))

    x_trans = data_trans(x)
    print(x_trans)

    # x = torch.tensor([[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
    #                   [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]]).unsqueeze(0)
    # x = x.permute(0, 3, 1, 2)
    #
    # x_trans = data_trans(x, [[1, 5], [2, 6]])
    # print(x_trans[0])

    # x = torch.randn(1, 145, 1, 1)
    # x_trans = data_trans(x)
    # print(x_trans.shape)

    # x = torch.tensor([1, 2, 3, 4, 5, 6, 7]).unsqueeze(1).unsqueeze(1)
    # x = x.repeat(1, 2, 2)
    #
    # x_trans = data_trans(x)
    # print(x_trans)
