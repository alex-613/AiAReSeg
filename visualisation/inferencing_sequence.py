import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

# In this script we will visualise the results after we pass the sequences through the AiA track framework
# The inferencing from the model will not be done in real time, we will implement that in another script instead
# In this script we will simply specify a sequence, and the visualise the bounding boxes one by one

def get_names(root_path, seq_no='51'):

    """
    Grabs the names of of all of the files under a root path
    """

    root_path = _join_paths(root_path,seq_no)

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

def open_bbox(bbox_path, seq_no='51'):

    """
    Outputs a list full of the ground truth bounding boxes for each of the images of the sequence
    """

    bbox_path = os.path.join(bbox_path,f'Catheter-{seq_no}.txt')
    gt_list = []

    gt_file = open(bbox_path)
    for line in gt_file:
        line_list = line.split(',')
        line_list[-1] = line_list[-1].strip()
        gt_list.append(line_list)


    return gt_list

def open_orig_gt(root_path,seq_no):
    """
    Reads the ground truth file from the sequence
    """

    gt_path = os.path.join(root_path,"Catheter",f"Catheter-{seq_no}",f"gt_Catheter-{seq_no}.txt")
    gt_orig_list = []
    gt_orig_file = open(gt_path)
    for line in gt_orig_file:
        line_list = line[2:-3].split(',')
        gt_orig_list.append(line_list)

    return gt_orig_list

def loop_images(root_path, paths, bbox_path, seq_no):

    # Load the ground truth
    gt_list = open_bbox(bbox_path, seq_no)
    gt_list_orig = open_orig_gt(root_path,seq_no)

    # Once loaded, we can loop through the lines one by one and then plot the ground truth onto the image

    for index, path in enumerate(paths):
        im = Image.open(path)
        box = gt_list[index]
        gt_box = gt_list_orig[index]

        plot_results(im,box,gt_box,index)

def plot_results(pil_img, box,gt_box,index):

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    colors = COLORS * 100

    plt.figure(figsize=(10,10))
    plt.imshow(pil_img)

    ax = plt.gca()


    cx = int(box[0])
    cy = int(box[1])
    w = int(box[2])
    h = int(box[3])

    cx_gt = float(gt_box[0])
    cy_gt = float(gt_box[1])
    w_gt = float(gt_box[2])
    h_gt = float(gt_box[3])

    # Convert the bounding boxes into the xminymin, yminymax data style
    xmin, ymin, xmax, ymax = box_cxcywh_to_xyxy(cx, cy, w, h)
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = box_cxcywh_to_xyxy(cx_gt, cy_gt, w_gt, h_gt)

    ax.add_patch(plt.Rectangle((cx, cy), w, h, fill=False, color = colors[0], linewidth=3))
    # ax.text(cx, cy, "Model", fontsize=15,
    #         bbox=dict(facecolor='yellow', alpha=0.5))
    ax.add_patch(plt.Rectangle((cx_gt, cy_gt), w_gt, h_gt, fill=False, color = colors[1], linewidth=3))
    # ax.text(cx_gt, cy_gt, "Ground Truth", fontsize=15,
    #         bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()
    # plt.savefig(f"/home/atr17/Desktop/AiA results/catheter results experiment 4.1/750/{index}.png")
    # plt.close()

def box_cxcywh_to_xyxy(x_c, y_c, w, h):

    xmin = x_c - 0.5 * w
    ymin = y_c - 0.5 * h
    xmax = x_c + 0.5 * w
    ymax = y_c + 0.5 * h

    return xmin, ymin, xmax, ymax

def compute_evaluation_metrics(gt_list, gt_orig_list):
    IOU_list = []
    MAE_list = []
    for box, gt_box in zip(gt_list,gt_orig_list):
        IOU_list.append(compute_IOU(box,gt_box))
        MAE_list.append(compute_MAE(box,gt_box))

    return np.mean(IOU_list), np.mean(MAE_list)

def compute_IOU(box, gt_box):
    # Intersection over union metric
    box = [round(float(i)) for i in box]
    gt_box = [round(float(i)) for i in gt_box]

    xmin, ymin, xmax, ymax = box_cxcywh_to_xyxy(box[0], box[1], box[2], box[3])
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = box_cxcywh_to_xyxy(gt_box[0], gt_box[1], gt_box[2], gt_box[3])

    x_inter1 = max(xmin,xmin_gt)
    y_inter1 = min(ymax,ymax_gt)
    x_inter2 = min(xmax,xmax_gt)
    y_inter2 = max(ymin,ymin_gt)

    width_inter = x_inter2-x_inter1
    height_inter = y_inter1-y_inter2

    if width_inter > 0 and height_inter > 0:
        intersect = width_inter * height_inter
        union = box[2] * box[3] + gt_box[2] * gt_box[3] - intersect
        return intersect / union

    else:
        intersect = 0
        return 0

def compute_MAE(box,gt_box):
    # Compute the size of im
    W,H = (800,600)

    box = [float(b) for b in box]
    gt_box = [float(b) for b in gt_box]

    box_mask = torch.zeros((W,H))
    gt_box_mask = torch.zeros((W,H))

    # Then we convert the mask into the masks
    box_mask[round(box[0]-0.5*box[2]):round(box[0]+0.5*box[2]),round(box[1]-0.5*box[3]):round(box[1]+0.5*box[3])] = 1
    gt_box_mask[round(gt_box[0] - 0.5 * gt_box[2]):round(gt_box[0] + 0.5 * gt_box[2]), round(gt_box[1] - 0.5 * gt_box[3]):round(gt_box[1] + 0.5 * gt_box[3])] = 1

    criterion = torch.nn.L1Loss()

    mae = criterion(box_mask,gt_box_mask)

    return mae


def compute_mAP():
    pass

def compute_AUC():
    pass


if __name__ == '__main__':
    bbox_path = '/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiATrack/AiATrack/tracking/PATH/AiATrack/test/tracking_results/aiatrack/baseline'
    root_path = '/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/Final_Dataset_Rot/Images/Val_2'

    # seq_no = '925'
    #
    # names, paths = get_names(root_path,seq_no)
    #
    # names = sorted(names)
    # paths = sorted(paths)
    # #
    # #print(paths)
    #
    # gt_list = open_bbox(bbox_path, seq_no)
    # gt_orig_list = open_orig_gt(root_path, seq_no)
    #
    # #
    # # print(gt_list)
    # # print(gt_orig_list)
    #
    # #loop_images(root_path, paths, bbox_path, seq_no)
    #
    # # TODO: Should also implement the ground truth visualisation when got time.
    # average_IOU = compute_evaluation_metrics(gt_list,gt_orig_list)
    #
    # print(average_IOU)

    # Loop through all sequence numbers from 690 to 921
    dataset_IOU = []
    dataset_MAE = []
    for i in range(690,922):
        seq_no = str(i)
        names, paths = get_names(root_path, seq_no)

        names = sorted(names)
        paths = sorted(paths)
        gt_list = open_bbox(bbox_path, seq_no)
        gt_orig_list = open_orig_gt(root_path, seq_no)

        average_IOU, average_MAE = compute_evaluation_metrics(gt_list, gt_orig_list)


        dataset_IOU.append(average_IOU)
        dataset_MAE.append(average_MAE)

    print("Mean IOU for Catheter Dataset:")
    print(np.mean(dataset_IOU))

    print("Mean MAE for Catheter Dataset:")
    print(np.mean(dataset_MAE))







