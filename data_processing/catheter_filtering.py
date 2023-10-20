# In this script we experiment with opencv to see whether we can filter out the catheter from the segmentation of the aorta

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans
import monai
import torch
import math
import re


def get_names(root_path, mode="Val"):

    """
    Grabs the names of of all of the files under a root path
    """

    images_root = os.path.join(root_path, "Images", "Val", "Catheter")
    masks_root = os.path.join(root_path, "Masks", "Val")

    image_seq_folders = os.listdir(images_root)
    mask_seq_folders = os.listdir(masks_root)

    image_list = []
    mask_list = []
    for image_seq_folder,mask_seq_folders in zip(image_seq_folders,mask_seq_folders):
        image_seq_path = os.path.join(images_root, image_seq_folder)
        mask_seq_path = os.path.join(masks_root, mask_seq_folders)

        images = os.listdir(image_seq_path)
        masks = os.listdir(mask_seq_path)

        for image,mask in zip(images, masks):
            image_path = os.path.join(image_seq_path, image)
            mask_path = os.path.join(mask_seq_path, mask)

            image_list.append(image_path)
            mask_list.append(mask_path)

    return image_list,mask_list

def get_masks(root_path, mode="Val"):

    """
    Grabs the names of of all of the files under a root path
    """

    masks_root = root_path

    mask_seq_folders = os.listdir(masks_root)

    image_list = []
    mask_list = []
    for mask_seq_folders in mask_seq_folders:
        mask_seq_path = os.path.join(masks_root, mask_seq_folders)

        masks = os.listdir(mask_seq_path)

        for mask in masks:
            mask_path = os.path.join(mask_seq_path, mask)

            mask_list.append(mask_path)

    return mask_list


def apply_aorta_mask(image, mask):

    return image*mask

def apply_thresholding(image):

    """
    Image: A numpy array read in by opencv
    """
    threshold = 0.6
    unique_intensities = np.unique(image)
    len_intensities = len(unique_intensities)
    thresholding_intensity = unique_intensities[math.floor(threshold*len_intensities)]

    image = np.where(image < thresholding_intensity, 0, image)
    image_sum = np.sum(image)

    return image

def convert_image_to_coordinates(image):

    image = image[:,:,0]

    cols, rows = np.where(image>0)

    cols = cols[:, np.newaxis]
    rows = rows[:, np.newaxis]

    coord = np.concatenate((cols, rows), axis=1)

    return coord

def apply_clustering(X):


    kmeans = KMeans(n_clusters=2)
    r,c = X.shape
    if r == 0 or c == 0:
        print("Please check")
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    return X,y_pred

def convert_coordinates_to_mask(coord,H,W):
    mask = np.zeros(shape=(H,W),dtype=np.uint8)

    cols = coord[:,0]
    rows = coord[:,1]

    for col,row in zip(cols,rows):
        if col>W:
            col = W
        if row>H:
            row = H
        mask[col-1,row-1] = 1
    return mask

def read_image(image_list, mask_list, gt_mask_list, transforms=[]):

    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    dice_list = []
    mae_list = []
    dice = monai.losses.DiceLoss()
    mae = monai.metrics.MAEMetric()
    counter = 0
    for image_path, mask_path, gt_mask_path in zip(image_list,mask_list,gt_mask_list):
        # For each image mask pair, lets filter and plot first
        image = cv2.imread(image_path)
        H,W,_ = image.shape
        mask = cv2.imread(mask_path)
        gt_mask = cv2.imread(gt_mask_path)
        gt_mask_catheter = np.where(gt_mask == 2,1,0)
        mask = np.where(mask==1, [0,255,0], [0,0,0])

        seg_image = apply_aorta_mask(image, gt_mask)

        if 'thresholding' in transforms:
            seg_image = apply_thresholding(seg_image)

        if 'clustering' in transforms:
            coord = convert_image_to_coordinates(seg_image)
            [X, y_pred] = apply_clustering(coord)
            unique_labels = np.unique(y_pred)
            X_dict = {}
            for unique_label in unique_labels:
                X_dict[f'{unique_label}'] = {}
                X_dict[f'{unique_label}']['data'] = X[y_pred == unique_label]
                variance = np.var(X[y_pred == unique_label], axis=0)
                X_dict[f'{unique_label}']['variance_rms'] = np.sqrt(np.mean(np.sum(variance)**2))

            min_variance = 10000
            variance_key = ''
            for X_key in X_dict.keys():
                if X_dict[X_key]['variance_rms'] < min_variance:
                    min_variance = X_dict[X_key]['variance_rms']
                    variance_key = X_key

        colored_mask = np.where(gt_mask_catheter==1,[255,0,0],[0, 0, 0]).astype(float)
        image = image.astype(float)/255.0
        image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

        # fig, axs = plt.subplots(1,2,figsize=(20,20))
        #
        # axs[0].imshow(image)
        # axs[1].imshow(colored_mask)
        # axs[0].scatter(X_dict[variance_key]['data'][:,1], X_dict[variance_key]['data'][:,0])
        #
        # plt.show()

        # save_dir = f'/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiASeg_w_temp/AiATrack/tracking/PATH/AiATrack/test/tracking_results/aiatrack/images_clustering/{counter}.png'

        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # plt.savefig(save_dir)
        # plt.close()

        model_mask_catheter = convert_coordinates_to_mask(X_dict[f'{variance_key}']['data'],H=H,W=W)

        # plt.figure()
        # plt.imshow(model_mask_catheter)
        # plt.show()
        # model_mask_catheter_plot = np.repeat(model_mask_catheter[:,:,np.newaxis],3, axis=2)
        # model_mask_catheter_plot = np.where(model_mask_catheter_plot==1,[0,255,0],[0,0,0]).astype(float)/255.0
        # # gt_mask_catheter = np.where(gt_mask_catheter == 1, [255, 0, 0], [0, 0, 0])
        # image = cv2.addWeighted(model_mask_catheter_plot, 1, colored_mask, 0.5, 0)

        # fig, axs = plt.subplots(1,2,figsize=(20,20))
        #
        # axs[0].imshow(image)
        # axs[1].imshow(model_mask_catheter_plot)
        # axs[0].scatter(X_dict[variance_key]['data'][:, 1], X_dict[variance_key]['data'][:, 0])
        #
        # plt.show()

        gt_tensor = torch.tensor(gt_mask_catheter[:,:,0])
        model_tensor = torch.tensor(model_mask_catheter)
        metric = dice(gt_tensor,model_tensor)
        mae_metric = mae(gt_tensor,model_tensor)
        # print(metric)
        # print(mae_metric)
        dice_list.append(metric)
        mae_list.append(torch.sum(mae_metric))
        #print(dice_list)
        #print(mae_list)
        counter += 1

    average_dice = np.mean(dice_list)
    average_mae = np.mean(mae_list)
    print(average_dice)
    print(average_mae)

def custom_sort_key(file_name):
    # Split the file name into folder and photo number
    folder_name, photo_number = re.match(r'(.+?)/(\d+)\.png', file_name).groups()
    return (folder_name, int(photo_number))


if "__main__" == __name__:

    print("Running filter")
    root_path = "/media/atr17/HDD Storage/US_axial_data_shallower_V2"
    model_mask_path = "/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiASeg_w_temp/AiATrack/tracking/PATH/AiATrack/test/tracking_results/aiatrack/AiASeg_w_temp_cath_dilate"
    image_list, gt_mask_list = get_names(root_path)
    model_mask_list = get_masks(model_mask_path)

    image_list = sorted(image_list)
    gt_mask_list = sorted(gt_mask_list)
    #model_mask_list = sorted(model_mask_list)
    model_mask_list = sorted(model_mask_list,key=custom_sort_key)

    read_image(image_list=image_list, mask_list=model_mask_list, gt_mask_list=gt_mask_list, transforms=['thresholding', 'clustering'])
