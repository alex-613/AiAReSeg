import os
import shutil


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

            new_mask_path = os.path.join(masks_folder,mask_seq, image_file)

            # Rename the mask file to match the image file
            os.rename(mask_path,new_mask_path)


# Example usage:
images_folder = '/media/atr17/HDD Storage/US_axial_data_shallower/Images'
masks_folder = '/media/atr17/HDD Storage/US_axial_data_shallower/Masks'
rename_mask_files(images_folder, masks_folder)
