import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
import shutil

def loop_paths(root):
    """
    Grabs the names of of all of the files under a root path
    """

    names = []
    paths = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            # print(os.path.join(path, name))
            # print(name)
            file_dir = os.path.join(os.path.basename(path),name)
            names.append(file_dir)
            paths.append(os.path.join(path, name))

    return names, paths


def flip_image(path):

    for file in os.listdir(path):
        img_path = os.path.join(path,file)
        img = cv2.imread(img_path)

        img_flipped = cv2.flip(img, 0)
        cv2.imwrite(img_path, img_flipped)

        # cv2.imshow("Vertical Flip",img_flipped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def renaming_images(root_path):

    for images in os.listdir(root_path):
        old_name = images
        new_name = images[:-8] + images[-7:]
        old_path = os.path.join(root_path, old_name)
        new_path = os.path.join(root_path, new_name)
        os.rename(old_path, new_path)

def renaming_images_imfusion_file(root_path):

    all_files = sorted(os.listdir(root_path))
    for images in all_files:
        old_name = images

        name_no_suf = images[:-4]
        final_index = name_no_suf.split("_")[-1]

        if len(final_index) == 1:
            new_name = name_no_suf + "_000" + ".png"

        elif len(final_index) > 1:
            new_name = name_no_suf[:-4] + "_" + name_no_suf[-3:] + ".png"

        old_path = os.path.join(root_path, old_name)
        new_path = os.path.join(root_path, new_name)
        os.rename(old_path, new_path)

def renaming_masks_imfusion_file(root_path):

    all_files = sorted(os.listdir(root_path))
    for images in all_files:
        old_name = images

        name_no_suf = images[:-4]
        final_index = name_no_suf.split("_")[-1]

        if len(final_index) == 0:
            new_name = name_no_suf + "000" + ".png"

        elif len(final_index) > 3:
            new_name = name_no_suf[:-4] + name_no_suf[-3:] + ".png"

        else:
            new_name = old_name

        old_path = os.path.join(root_path, old_name)
        new_path = os.path.join(root_path, new_name)
        os.rename(old_path, new_path)

def remove_sweep_from_name(root_path):
    # There is an extra "sweep" infront of every image file, rename it so it is removed
    all_files = sorted(os.listdir(root_path))
    for images in all_files:
        old_name = images

        name_no_suf = images[:-4]
        name_no_first_str = name_no_suf.split("_")[1:]
        new_name = "_".join(name_no_first_str) + ".png"

        old_path = os.path.join(root_path, old_name)
        new_path = os.path.join(root_path, new_name)
        os.rename(old_path, new_path)

def remove_extra_zero(root_path):

    # There is an extra "sweep" infront of every image file, rename it so it is removed
    all_files = sorted(os.listdir(root_path))
    for images in all_files:
        old_name = images

        name_no_suf = images[:-4]
        second_last_str = name_no_suf.split("_")[-2]

        if len(second_last_str) == 2:
            new_name = name_no_suf[:-5] + name_no_suf[-4:] + ".png"
        else:
            new_name = old_name

        old_path = os.path.join(root_path, old_name)
        new_path = os.path.join(root_path, new_name)
        os.rename(old_path, new_path)

def filtering_no_catheter(root_path,root_destination):

    mask_path = os.path.join(root_path,"Masks")
    image_path = os.path.join(root_path,"Images")

    mask_destination = os.path.join(root_destination, "Masks")
    image_destination = os.path.join(root_destination, "Images")

    for path in os.listdir(image_path):
        file_path = os.path.join(mask_path,path)
        file_path_img = os.path.join(image_path,path)

        numpydata = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        catheter_mask_rows, catheter_mask_cols = np.where(numpydata==2)
        aorta_mask_rows, aorta_mask_cols = np.where(numpydata==1)

        if len(catheter_mask_cols)>0:
            destination_mask = os.path.join(mask_destination,path)
            destination_img = os.path.join(image_destination,path)
            shutil.copy(file_path,destination_mask)
            shutil.copy(file_path_img,destination_img)

    return 0

def splitting(root_path):
    mask_path = os.path.join(root_path,"Masks")
    image_path = os.path.join(root_path,"Images")

    all_masks = sorted(os.listdir(mask_path))

    for file in all_masks:
        # Grab the generic names of the files
        special_identifier = file[-15:-8]

        folder_path_image = os.path.join(image_path,special_identifier)
        folder_path_mask = os.path.join(mask_path,special_identifier)

        if not os.path.exists(folder_path_image):
            os.mkdir(folder_path_image)
        if not os.path.exists(folder_path_mask):
            os.mkdir(folder_path_mask)

        mask_file_old = os.path.join(mask_path,file)
        image_file_old = os.path.join(image_path,file)

        mask_file_new = os.path.join(folder_path_mask,file)
        image_file_new = os.path.join(folder_path_image,file)

        shutil.move(mask_file_old, mask_file_new)
        shutil.move(image_file_old, image_file_new)

def reordering_images(root_path):

    mask_path = os.path.join(root_path,"Masks")
    image_path = os.path.join(root_path,"Images")

    all_image_folders= sorted(os.listdir(mask_path))
    all_mask_folders= sorted(os.listdir(image_path))

    # Start a loop for each folder
    for image_folder,mask_folder in zip(all_image_folders,all_mask_folders):
        all_images_path = os.path.join(image_path, image_folder)
        all_masks_path = os.path.join(mask_path, mask_folder)
        all_images = sorted(os.listdir(all_images_path))
        all_masks = sorted(os.listdir(all_masks_path))

        counter = 0

        for image,mask in zip(all_images,all_masks):
            old_image_name = os.path.join(image_path, image_folder,image)
            old_mask_name = os.path.join(mask_path, mask_folder,mask)

            str_counter = str(counter)

            if len(str_counter) == 1:
                new_image_name = old_image_name[:-7] + "00" + str_counter + ".png"
                new_mask_name = old_mask_name[:-7] + "00" + str_counter + ".png"

            elif len(str_counter) == 2:
                new_image_name = old_image_name[:-7] + "0" + str_counter + ".png"
                new_mask_name = old_mask_name[:-7] + "0" + str_counter + ".png"

            elif len(str_counter) == 3:
                new_image_name = old_image_name[:-7] + str_counter + ".png"
                new_mask_name = old_mask_name[:-7] + str_counter + ".png"

            os.rename(old_image_name,new_image_name)
            os.rename(old_mask_name, new_mask_name)

            counter += 1

def renaming_folders(root_path):

    mask_path = os.path.join(root_path,"Masks")
    image_path = os.path.join(root_path,"Images")

    all_image_folders= sorted(os.listdir(mask_path))
    all_mask_folders= sorted(os.listdir(image_path))

    counter = 760
    # Start a loop for each folder
    for image_folder,mask_folder in zip(all_image_folders,all_mask_folders):
        old_images_folder_path = os.path.join(image_path, image_folder)
        old_masks_folder_path = os.path.join(mask_path, mask_folder)

        new_folder_name = f"Catheter-{counter}"
        new_images_folder_path = os.path.join(image_path, new_folder_name)
        new_masks_folder_path = os.path.join(mask_path, new_folder_name)

        os.rename(old_images_folder_path, new_images_folder_path)
        os.rename(old_masks_folder_path, new_masks_folder_path)

        counter += 1





if __name__ == '__main__':
    # path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Catheter_Simulations/Catheter_Simulaton_031/us_simulation/rotations/Masks"
    # flip_image(path)

    # root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/Catheter_simulation_031/Images"
    # renaming_images_imfusion_file(root_path)

    # root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/Catheter_simulation_031/Masks"
    # renaming_masks_imfusion_file(root_path)

    # root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/new_axial_data/Catheter_simulation_031/masks"
    # renaming_images(root_path)

    # root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/Catheter_simulation_031/Images"
    # remove_sweep_from_name(root_path)

    # root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/Catheter_simulation_031/Images"
    # remove_extra_zero(root_path)

    # root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/Catheter_simulation_009"
    # root_destination="/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/final/Catheter_simulation_009"
    # filtering_no_catheter(root_path, root_destination)

    # root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/final/Catheter_simulation_001"
    # splitting(root_path=root_path)

    # root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/final/Catheter_simulation_001"
    # reordering_images(root_path)

    root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/final/Catheter_simulation_031"
    renaming_folders(root_path)