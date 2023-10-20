import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms

# In this script we will visualise the results after we pass the sequences through the AiA track framework
# The inferencing from the model will not be done in real time, we will implement that in another script instead
# In this script we will simply specify a sequence, and the visualise the bounding boxes one by one


def get_names(root_path, seq_name=''):

    """
    Grabs the names of of all of the files under a root path
    """

    # Split the sequence names in order to get the sequence class

    root_path = _join_paths(root_path,seq_name)

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

def _join_paths(root_path, seq_name=''):

    """
    Joins the paths of the root with the specified sequence
    """

    # Split the sequence name in order to get the sequence class
    class_name = seq_name.split('-')[0]

    return os.path.join(root_path,f'{class_name}',f'{seq_name}','img')

def open_bbox(bbox_path, seq_name='' ):

    """
    Outputs a list full of the ground truth bounding boxes for each of the images of the sequence
    """

    gt_path = os.path.join(bbox_path, f'{seq_name}.txt')
    gt_list = []

    gt_file = open(gt_path)
    for line in gt_file:
        line_list = line.split(',')
        line_list[-1] = line_list[-1].strip()
        gt_list.append(line_list)

    return gt_list

def box_cxcywh_to_xyxy(x_c, y_c, w, h):

    xmin = x_c - 0.5 * w
    ymin = y_c - 0.5 * h
    xmax = x_c + 0.5 * w
    ymax = y_c + 0.5 * h

    return xmin, ymin, xmax, ymax

def loop_images(paths, bbox_path, seq_name,object):

    # Load the ground truth
    box_list = open_bbox(bbox_path, seq_name)
    gt_list_orig = open_orig_gt(root_path, seq_no, object)

    # Once loaded, we can loop through the lines one by one and then plot the ground truth onto the image

    for index, path in enumerate(paths):
        im = Image.open(path)
        box = box_list[index]
        gt_box = gt_list_orig[index]

        plot_results(im, box,gt_box,index)

def open_orig_gt(root_path,seq_no,object):
    """
    Reads the ground truth file from the sequence
    """

    gt_path = os.path.join(root_path,f"{object}",f"{object}-{seq_no}",f"groundtruth.txt")
    gt_orig_list = []
    gt_orig_file = open(gt_path)
    for line in gt_orig_file:
        line_list = line.split(',')
        gt_orig_list.append(line_list)

    return gt_orig_list


def plot_results(pil_img, box,gt_bbox,index):

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    colors = COLORS * 100

    transform = transforms.ToTensor()
    image = transform(pil_img)

    orig_size = image.shape

    image = transforms.functional.resize(img=image,
                                         size=(320, 320)).permute(1, 2, 0).numpy()

    new_size = image.shape

    plt.figure(figsize=(16,10))
    plt.imshow(image)

    # ax = plt.gca()

    x_factor = 320/orig_size[2]
    y_factor = 320/orig_size[1]

    # plt.figure(figsize=(16,10))
    # plt.imshow(pil_img)

    ax = plt.gca()

    cx = round(float(box[0])*x_factor)
    cy = round(float(box[1])*y_factor)
    w = round(float(box[2])*x_factor)
    h = round(float(box[3])*y_factor)

    cx_gt = round(float(gt_bbox[0])*x_factor)
    cy_gt = round(float(gt_bbox[1])*y_factor)
    w_gt = round(float(gt_bbox[2])*x_factor)
    h_gt = round(float(gt_bbox[3])*y_factor)

    # Convert the bounding boxes into the xminymin, yminymax data style
    xmin, ymin, xmax, ymax = box_cxcywh_to_xyxy(cx, cy, w, h)

    ax.add_patch(plt.Rectangle((cx, cy), w, h, fill=False, color = colors[0], linewidth=3))
    ax.add_patch(plt.Rectangle((cx_gt, cy_gt), w_gt, h_gt, fill=False, color=colors[1], linewidth=3))


    plt.axis('off')
    plt.savefig(f'/home/atr17/Desktop/CathFlow Results/bear-2-aia/{index}.png')
    #plt.show()
    plt.close()
def compute_evaluation_metrics(gt_list, gt_orig_list):
    IOU_list = []

    for box, gt_box in zip(gt_list,gt_orig_list):
        IOU_list.append(compute_IOU(box,gt_box))

    return np.mean(IOU_list)

def compute_IOU(box, gt_box):
    # Intersection over union metric
    box = [round(float(i)) for i in box]
    try:
        gt_box = [round(int(i)) for i in gt_box]
    except ValueError:
        print('gt-box error')

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


if __name__ == '__main__':
    bbox_path = '/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiATrack/AiATrack/tracking/PATH/AiATrack/test/tracking_results/aiatrack/LaSOT'
    root_path = '/media/atr17/HDD Storage/Datasets_Download/LaSOT/LaSOT'

    IOU_list = []

    # Grab the sequence names:
    for path, subdirs, files in os.walk(bbox_path):
        for name in files:
            # Loop through every single text file you have in the folder
            seq_name = name[:-4]

            seq_no = seq_name.split('-')[1]
            object = seq_name.split('-')[0]

            names, paths = get_names(root_path, seq_name)

            names = sorted(names)
            paths = sorted(paths)
            #
            #print(paths)

            gt_list = open_bbox(bbox_path, seq_name)

            orig_gt_list = open_orig_gt(root_path, seq_no, object)

            #print(gt_list)

            # loop_images(paths, bbox_path, seq_name,object)

            average_IOU = compute_evaluation_metrics(gt_list, orig_gt_list)

            #print(average_IOU)
            IOU_list.append(average_IOU)

    mean_IOU = np.mean(IOU_list)
    print(mean_IOU)




    # seq_name = 'bear-6'
    #
    # seq_no = seq_name.split('-')[1]
    # object = seq_name.split('-')[0]
    #
    # names, paths = get_names(root_path, seq_name)
    #
    # names = sorted(names)
    # paths = sorted(paths)
    # #
    # print(paths)
    #
    # gt_list = open_bbox(bbox_path, seq_name)
    #
    # orig_gt_list = open_orig_gt(root_path, seq_no, object)
    #
    # print(gt_list)
    #
    # #loop_images(paths, bbox_path, seq_name,object)
    #
    # average_IOU = compute_evaluation_metrics(gt_list,orig_gt_list)
    # # #
    # print(average_IOU)
    #
    # # TODO: Should also implement the ground truth visualisation when got time.
