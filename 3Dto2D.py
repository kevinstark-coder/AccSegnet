#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:10:08 2022

@author: ubuntu
"""
import os.path
import numpy as np
import cv2
import nibabel

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    '''if(len([min_idx]) == 1):
        output = volume[np.ix_(range(min_idx, max_idx + 1))]'''
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0]),
                               range(min_idx[1], max_idx[1]),
                               range(min_idx[2], max_idx[2]))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output
def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    '''if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))'''
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []

    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    '''for i in range(1, len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)'''
    '''idx_min = indxes[0].min()
    idx_max = indxes[0].max()'''
    r = 1/3
    a = idx_min[0]
    b = idx_max[0]
    #if(margin == 30):
    idx_min[0] = int((b-a)*r+a)
    idx_max[0] = int(b-(b-a)*r)
    '''else:
        idx_min[0] = int((b-a)*r+a)'''
    return idx_min, idx_max

def load_nifty_volume_as_array2(filename, with_header = False, is_label=False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data


dir_image = '/home/ubuntu/data/CHAOS/MR/image'
dir_target = '/home/ubuntu/data/CHAOS/MR/label'
save_image = '/media/ubuntu/name/2Dchaos/image_mr'
save_target = '/media/ubuntu/name/2Dchaos/label_mr'
value = 63
amargin = 30
paths_A = os.listdir(dir_image)
bounding = []
indexs = [0]

for dir in paths_A:
    target_path = os.path.join(dir_target,dir)
    target_img = load_nifty_volume_as_array2(target_path)
    target_img[target_img == value] = 255 #6
    target_img[target_img!=255] = 0
    boundingbox = get_ND_bounding_box(target_img, amargin)
    bounding.append(boundingbox)
    idex = boundingbox[1][0]-boundingbox[0][0]+indexs[-1]
    indexs.append(idex)
for index in range(0, indexs[-1]):
    id_A = int(np.where((np.array(indexs)) <= index)[0].max())
    A_path = os.path.join(dir_image, paths_A[id_A])
    slice_A = index-indexs[id_A]
    volume_A = load_nifty_volume_as_array2(A_path)
    volume_A = crop_ND_volume_with_bounding_box(volume_A,bounding[id_A][0],bounding[id_A][1])
    A_img = volume_A[slice_A]
    Atarget_path = os.path.join(dir_target,paths_A[id_A])
    volume_Atarget = load_nifty_volume_as_array2(Atarget_path)
    volume_Atarget = crop_ND_volume_with_bounding_box(volume_Atarget,bounding[id_A][0],bounding[id_A][1])
    Atarget_img = volume_Atarget[slice_A]
    Atarget_img[Atarget_img==63] = 255
    Atarget_img[Atarget_img<255] = 0
    '''A_img[A_img>240] = 240
    A_img[A_img<-160] = -160'''
    A_img = (A_img-A_img.min())/(A_img.max()-A_img.min())*255.0
    #Atarget_img = (Atarget_img-Atarget_img.min())/(Atarget_img.max()-Atarget_img.min())*255.0
    
    A_img = A_img.astype(np.float32)

    Atarget_img = Atarget_img.astype(np.float32)
    save_img_path = os.path.join(save_image, str(index+3000)+'.png') 
    save_target_path = os.path.join(save_target, str(index+3000)+'.png')
    cv2.imwrite(save_img_path, A_img)
    cv2.imwrite(save_target_path, Atarget_img)

