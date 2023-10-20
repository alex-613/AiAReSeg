import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_names(root_path, seq_no='51'):

    """
    Grabs the names of of all of the files under a root path
    """

    #root_path = _join_paths(root_path,seq_no)

    # dir_list = os.listdir(root_path)
    names = []
    paths = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            # print(os.path.join(path, name))
            # print(name)
            file_dir = os.path.join(os.path.basename(path),name)
            names.append(file_dir)
            paths.append(os.path.join(path, name))

    return names, paths




def open_mask(mask_path, seq_no='51'):

    """
    Outputs a list full of the ground truth bounding boxes for each of the images of the sequence
    """

    #mask_folder_path = os.path.join(mask_path,f'Catheter-{seq_no}','img')
    mask_folder_path = os.path.join(mask_path, f'Catheter-{seq_no}')
    names, files = get_names(mask_folder_path)
    mask_list = []
    files = sorted(files)
    for file in files:
        mask = torch.tensor(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        mask_list.append(mask)

    return torch.stack([mask for mask in mask_list], dim=0)

def open_images(image_path, seq_no='51'):

    #image_folder_path = os.path.join(image_path, 'Catheter', f'Catheter-{seq_no}', 'img')
    image_folder_path = os.path.join(image_path, 'Catheter', f'Catheter-{seq_no}')
    names, files = get_names(image_folder_path)
    image_list = []
    files = sorted(files)
    for file in files:
        image = torch.tensor(cv2.imread(file, cv2.IMREAD_COLOR))
        image_list.append(image)

    return torch.stack([image for image in image_list], dim=0)

def loop_image(image_tensor, gt_mask_tensor):

    # Having three seperate tensors allows you to plot everything on top of one another by simply doing one loop

    B,H,W,C = image_tensor.shape

    for i in range(B):
        image = image_tensor[i,...].float().numpy().astype(float)/255.0
        gt_mask = gt_mask_tensor[i,...].float()

        # Next, we need to plot the image and the masks

        fig, axs = plt.subplots(1,2,figsize=(20,20))

        # Plot the original image
        axs[0].imshow(image)
        # Plot the original image with the gt mask
        gt_mask = gt_mask.unsqueeze(2).repeat(1,1,3).numpy()
        gt_mask = np.where(gt_mask == 2, [255,0,0],[0,0,0]).astype(float) /255.0
        output1 = image
        output1 = cv2.addWeighted(output1, 1, gt_mask, 0.5, 0)
        axs[1].imshow(output1)
        plt.show()


        #plt.savefig(f'/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiATrack/AiATrack/tracking/PATH/AiATrack/test/tracking_results/aiatrack/images_for_catheters/91/{i}.png')
        #plt.close()


def plot_results(pil_img, box,gt_box, index):

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    colors = COLORS * 100

    plt.figure(figsize=(10,10))
    plt.imshow(pil_img)

    ax = plt.gca()

    plt.axis('off')
    plt.show()


def box_cxcywh_to_xyxy(x_c, y_c, w, h):

    xmin = x_c - 0.5 * w
    ymin = y_c - 0.5 * h
    xmax = x_c + 0.5 * w
    ymax = y_c + 0.5 * h

    return xmin, ymin, xmax, ymax

def loop_and_check():
    gt_mask_path = '/media/atr17/HDD Storage/US_axial_data_shallower/Masks/Val'
    image_path = "/media/atr17/HDD Storage/US_axial_data_shallower/Images/Val/Catheter"

    # Loop through each of the sequences and then the catheters, and record for me where there is no catheter

    mask_seqs = os.listdir(gt_mask_path)
    image_seqs = os.listdir(image_path)

    broken_masks = []

    for image_seq, mask_seq in zip(image_seqs, mask_seqs):
        image_seq_path = os.path.join(image_path, image_seq)
        mask_seq_path = os.path.join(gt_mask_path, mask_seq)

        images = sorted(os.listdir(image_seq_path))
        masks = sorted(os.listdir(mask_seq_path))

        for image, mask in zip(images, masks):
            image_file_path = os.path.join(image_seq_path, image)
            mask_file_path = os.path.join(mask_seq_path, mask)

            mask_array = cv2.imread(mask_file_path)

            if not 2 in np.unique(mask_array):
                broken_masks.append(mask_file_path)

    print(broken_masks)

if __name__ == '__main__':

    # # First lets load the masks
    # gt_mask_path = '/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Masks/Train'
    # image_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Images/Train"
    # gt_mask_tensor = open_mask(gt_mask_path, seq_no='160')
    # #model_mask_tensor = open_mask(model_mask_path, seq_no='58')
    #
    # image_tensor = open_images(image_path, seq_no='160')
    #
    # loop_image(image_tensor, gt_mask_tensor)
    #
    # print("Done")
    #
    # # TODO: Loop through each image, and then overlay the gt and the model outputs on the same image

    # # First lets load the masks
    gt_mask_path = '/media/atr17/HDD Storage/US_axial_data_shallower/Masks/Val'
    image_path = "/media/atr17/HDD Storage/US_axial_data_shallower/Images/Val"
    gt_mask_tensor = open_mask(gt_mask_path, seq_no='2')
    #model_mask_tensor = open_mask(model_mask_path, seq_no='58')

    image_tensor = open_images(image_path, seq_no='2')

    loop_image(image_tensor, gt_mask_tensor)

    print("Done")

    # loop_and_check()

    # TODO: Loop through each image, and then overlay the gt and the model outputs on the same image