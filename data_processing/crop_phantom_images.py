import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def sort_seq_names(name):
    parts = name.split("-")
    return int(parts[1])

def rename_mask_files(images_folder, masks_folder):

    image_sequences = os.listdir(images_folder)
    mask_sequences = os.listdir(masks_folder)

    image_sequences = sorted(image_sequences, key=sort_seq_names)
    mask_sequences = sorted(mask_sequences, key=sort_seq_names)

    for image_seq, mask_seq in zip(image_sequences, mask_sequences):

        image_seq_path = os.path.join(images_folder, image_seq)
        mask_seq_path = os.path.join(masks_folder, mask_seq)

        # Get a list of files in the images folder
        image_files = os.listdir(image_seq_path)
        mask_files = os.listdir(mask_seq_path)

        image_files = sorted(image_files)
        mask_files = sorted(mask_files)

        # Loop through the image files
        for image_file, mask_file in zip(image_files, mask_files):

            # Create the full paths for the image and mask files
            image_path = os.path.join(images_folder,image_seq, image_file)
            mask_path = os.path.join(masks_folder,mask_seq, mask_file)

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            cropped_img = image[45:535,:630,:]
            cropped_mask = mask[45:535, :630, :]

            # fig, axs = plt.subplots(1, 2, figsize=(20, 20))
            #
            # # Plot the original image
            # axs[0].imshow(image)
            # # Plot the original image with the gt mask
            # cropped_mask = np.where(cropped_mask == 2, [255, 0, 0], [0, 0, 0]).astype(float) / 255.0
            # output1 = cropped_img
            # output1 = cv2.addWeighted(output1, 1, cropped_mask, 0.5, 0)
            # axs[1].imshow(output1)
            #
            # plt.show()

            cv2.imwrite(image_path, cropped_img)
            cv2.imwrite(mask_path, cropped_mask)

images_folder = '/media/atr17/HDD Storage/US_axial_data_shallower/Images/Val/Catheter'
masks_folder = '/media/atr17/HDD Storage/US_axial_data_shallower/Masks/Val'
rename_mask_files(images_folder, masks_folder)