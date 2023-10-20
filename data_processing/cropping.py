# In file file we perform cropping of the ultrasound image

import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


# Write a function which grabs all of the masks and the image paths

def get_names(root_path):

    """
    Grabs the names of of all of the files under a root path
    """

    images_root = os.path.join(root_path, "Images", "Val", "Catheter")
    masks_root = os.path.join(root_path, "Masks","Val")

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

    # TODO: YOu need to sort this!
    return image_list, mask_list

def cropping(seg_image, seg_mask):

    cropped_image = seg_image[90:517, 35:670,:]
    cropped_mask = seg_mask[90:517, 35:670,:]

    return cropped_image, cropped_mask

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

def main(root):

    image_list, mask_list = get_names(root)

    for image_path, mask_path in zip(image_list, mask_list):

        # load the mask and the image

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        cropped_image, cropped_mask = cropping(image, mask)

        #plot_image(cropped_image, cropped_mask, alpha=0.5)

        cv2.imwrite(mask_path, cropped_mask)
        cv2.imwrite(image_path, cropped_image)

if __name__ == "__main__":
    root = "/media/atr17/HDD Storage/US_axial_data_shallower_V2"
    main(root)
