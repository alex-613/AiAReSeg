import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


# Write a function which grabs all of the masks and the image paths

def get_names(root_path, mode="Val"):

    """
    Grabs the names of of all of the files under a root path
    """

    images_root = os.path.join(root_path, "Images", mode, "Catheter")
    masks_root = os.path.join(root_path, "Masks", mode)

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

    return image_list, mask_list

def dilation_and_translation(seg_mask, kernel_size= 5, translation=3):


    # Define the dilation kernel (a square kernel of size 3x3 in this example)
    #kernel = np.ones(kernel_size, np.uint8)

    # Define the size of the kernel for dilation (adjust as needed)

    # # Create a circular kernel (you can use cv2.getStructuringElement as shown in a previous response)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    #
    # # Create masks for label 1 and label 2
    # label1_mask = (seg_mask == 1).astype(np.uint8)
    # label2_mask = (seg_mask == 2).astype(np.uint8)
    #
    # # Perform dilation on the label 2 mask
    # dilated_label2 = cv2.dilate(label2_mask, kernel, iterations=1)
    #
    # # Combine the original label 1 mask and the dilated label 2 mask
    # result_mask = label1_mask + dilated_label2

    orig_aorta_mask = (seg_mask==1).astype(np.uint8)

    hole_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    orig_aorta_mask = cv2.morphologyEx(orig_aorta_mask, cv2.MORPH_CLOSE, hole_kernel)

    # Perform dilation on the segmentation mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    dilated_mask = cv2.dilate(seg_mask, kernel, iterations=1)

    translation_matrix = np.float32([[1, 0, 0], [0, 1, -translation]])

    # Apply the translation to the mask
    mask = cv2.warpAffine(dilated_mask, translation_matrix, (dilated_mask.shape[1], dilated_mask.shape[0]))

    catheter_mask = (mask==2).astype(np.uint8)

    mask = catheter_mask + orig_aorta_mask

    return mask

def plot_image(image, mask, alpha=0.5):
    # Load the image and the segmentation mask

    # Resize the mask to match the dimensions of the image
    #mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Convert the mask to a 3-channel image (assuming it's a grayscale mask)
    # if mask.shape[-1] == 1:
    #     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    mask_plot = np.where(mask==2, [255,0,0], [0,0,0]).astype(float)/255.0
    image = image.astype(float)/255.0

    # Ensure alpha is in the valid range [0, 1]
    alpha = max(0.0, min(1.0, alpha))

    # Create the overlay by blending the image and mask
    overlay = cv2.addWeighted(image, 1, mask_plot, alpha, 0)

    plt.figure(figsize=(15,10))
    plt.imshow(overlay)
    plt.show()

def main(root,mode="Train"):

    image_list, mask_list = get_names(root, mode=mode)

    fix_mask = []
    for image_path, mask_path in zip(image_list, mask_list):

        # load the mask and the image

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)


        new_mask = dilation_and_translation(mask, kernel_size=10, translation=3)

        if not 2 in np.unique(new_mask):
            fix_mask.append(mask_path)

        #plot_image(image, new_mask, alpha=0.5)

        cv2.imwrite(mask_path, new_mask)

    print(fix_mask)

if "__main__" == __name__:

    root = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/new_axial_dataset_dilated2"
    main(root, mode="Val")












