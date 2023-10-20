import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import monai

# In this script we will visualise the results after we pass the sequences through the AiA track framework
# The inferencing from the model will not be done in real time, we will implement that in another script instead
# In this script we will simply specify a sequence, and the visualise the bounding boxes one by one

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

def _join_paths(root_path, seq_no='51'):

    """
    Joins the paths of the root with the specified sequence
    """

    return os.path.join(root_path,'Catheter',f'Catheter-{seq_no}','img')

def sort_seq_names(name):
    parts = name.split("/")
    return int(parts[-1][:-4])

def sort_numeric_part(name):
    # Split the filename into non-digit and digit parts
    name = name.split("/")[-1]

    # Extract the numeric part as an integer
    return int(name[:-4])

def open_mask(mask_path, seq_no='51'):

    """
    Outputs a list full of the ground truth bounding boxes for each of the images of the sequence
    """

    mask_folder_path = os.path.join(mask_path,f'Catheter-{seq_no}')

    names, files = get_names(mask_folder_path)
    mask_list = []
    final_elem = files[0].split('/')[-1]
    if len(final_elem)>5:
        files = sorted(files)
    else:
        files = sorted(files, key=sort_seq_names)


    for file in files:
        mask = torch.tensor(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        mask_list.append(mask)

    return torch.stack([mask for mask in mask_list], dim=0)

def open_model_mask(mask_path, seq_no='51'):
    mask_folder_path = os.path.join(mask_path,f'Catheter-{seq_no}')

    names, files = get_names(mask_folder_path)
    mask_list = []
    final_elem = files[0].split('/')[-1]

    files = sorted(files, key=sort_numeric_part)



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


def loop_image(image_tensor, gt_mask_tensor, model_mask_tensor, id):

    # Having three seperate tensors allows you to plot everything on top of one another by simply doing one loop

    B,H,W,C = image_tensor.shape

    for i in range(B):
        image = image_tensor[i,...].float().numpy().astype(float)/255.0 * 0.8
        gt_mask = gt_mask_tensor[i,...].float()
        model_mask = model_mask_tensor[i,...].float()


        # Next, we need to plot the image and the masks

        # fig, axs = plt.subplots(1,3,figsize=(20,20))
        #
        # # Plot the original image
        # axs[0].imshow(image)
        # # Plot the original image with the gt mask
        # gt_mask = gt_mask.permute(1,2,0).repeat(1,1,3).numpy()
        # gt_mask = np.where(gt_mask == 1, [0,255,0],[0,0,0]).astype(float) /255.0
        # output1 = image
        # output1 = cv2.addWeighted(output1, 1, gt_mask, 0.5, 0)
        # axs[1].imshow(output1)
        # # Plot the original image with the output mask
        # model_mask = model_mask.permute(1,2,0).repeat(1,1,3).numpy()
        # model_mask = np.where(model_mask == 1, [255,0,0],[0,0,0]).astype(float)/255.0
        # output2 = image
        # output2 = cv2.addWeighted(output2, 1, model_mask, 0.5, 0)
        # axs[2].imshow(output2)

        model_mask = model_mask.permute(1, 2, 0).repeat(1, 1, 3).numpy()
        model_mask = np.where(model_mask == 1, [255, 0, 0], [0, 0, 0]).astype(float) / 255.0
        output2 = image
        output2 = cv2.addWeighted(output2, 1, model_mask, 0.5, 0)
        plt.figure(figsize=(15,15))
        ax = plt.gca()
        plt.imshow(output2)
        ax.set_axis_off()
        #plt.show()


        save_dir = f'/home/atr17/Desktop/AiA results/AiAReSeg_catheter/{id}'
        #
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(os.path.join(save_dir,f"{i}.png"))
        plt.close()


if __name__ == '__main__':


    # First lets load the masks
    gt_mask_path = '/media/atr17/HDD Storage/US_axial_data_shallower_V2/Masks/Val'
    model_mask_path = '/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiASeg_w_temp/AiATrack/tracking/PATH/AiATrack/test/tracking_results/aiatrack/AiASeg_s+p'
    image_path = "/media/atr17/HDD Storage/US_axial_data_shallower_V2/Images/Val"

    dice = monai.losses.DiceLoss(jaccard=False)
    mae = monai.metrics.MAEMetric()

    dice_list = []
    mae_list = []
    for i in range(1, 6):
        seq_no = str(i)
        gt_mask_tensor = open_mask(gt_mask_path, seq_no=seq_no).unsqueeze(1)
        model_mask_tensor = open_model_mask(model_mask_path, seq_no=seq_no).unsqueeze(1)
        gt_mask_tensor = torch.where(gt_mask_tensor==2,1,0)
        image_tensor = open_images(image_path, seq_no=seq_no)

        dice_list.append(dice(gt_mask_tensor, model_mask_tensor).item())
        mae_list.append(torch.mean(mae(gt_mask_tensor, model_mask_tensor)).item())

        loop_image(image_tensor, gt_mask_tensor, model_mask_tensor, id=i)
    dice_list = np.array(dice_list)
    average_dice = np.mean(dice_list)
    mae_list = np.array(mae_list)
    average_mae = np.mean(mae_list)
    print("Done")
    print(f"Dice_loss:{average_dice}")
    print(f"MAE: {average_mae}")
    # Performance 0.0808, dice:0.9192 , average mae:0.002134
    # Catheter performance, dice: 0.831, average mae: 0.000138

    # TODO: Loop through each image, and then overlay the gt and the model outputs on the same image

