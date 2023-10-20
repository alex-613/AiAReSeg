import collections
import csv
import os
import os.path
import random
from collections import OrderedDict
import re
import numpy as np
import pandas
import pandas as pd
import torch

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from lib.train.data import opencv_loader
from lib.train.data import pil_loader
from .base_video_dataset import BaseVideoDataset
import cv2
import json
import os

class Catheter_tracking(BaseVideoDataset):

    def __init__(self, root, image_loader=opencv_loader, vid_ids=None, mode='Train'):
        """
        Args:
            root: Path to the catheter tracking = dataset.
            image_loader (jpeg4py_loader): The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                           is used by default.
            vid_ids: List containing the IDs of the videos used for training. Note that the sequence IDs are not always the same, there are different IDs for each of the patient images.
            split: If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                   vid_ids or split option can be used at a time.
            data_fraction: Fraction of dataset to be used. The complete dataset is used by default.
        """
        self.root = env_settings().catheter_tracking_dir if root is None else root
        #self.root = os.path.join(self.root, mode)

        self.image_loader = image_loader
        super().__init__('Catheter_tracking', self.root, self.image_loader)

        self.class_list = ['catheter']

        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        # Go through the entire directory, and then create the lists in the form of Patient_id_sequence id
        self.sequence_list = self._build_sequence_list()

        self.seq_per_class = self._build_class_list()


    # We will keep it simple and the datasplitting will be done manually instead of programatically
    def _build_sequence_list(self):
        # Try to access the directories in the root folder of the dataset and get the sequence list
        return self._get_train_sequences()

    def _get_train_sequences(self):
        # Loop through the subdirectory of the training folder to get all the names of the training sequences
        seq_list = []
        for names,subdires,files in os.walk(self.root):
            for subdir in subdires:
                if (subdir != 'img') and (subdir != 'Catheter'):
                    seq_list.append(subdir)
        seq_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        return seq_list
        #return sorted(seq_list)

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]
        #input(seq_per_class)
        return seq_per_class



    def get_name(self):
        return 'catheter_tracking'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def _read_bb_anno(self,seq_path):
        # For each of the folders of sequences, you need to generate the annotations and put them into an annotation file
        seq = seq_path.split('/')[-1]
        bb_anno_file = os.path.join(seq_path,f'gt_{seq}.txt')
        gt = []

        with open(bb_anno_file) as f:
            for line in f.readlines():
                line = line[2:-3]
                line = line.split(",")
                line = [float(i) for i in line]
                line = self.convert_cxcywh_2_x1y1wh(line)
                # Open up the file and convert it from cxcywh to x1y1wh

                gt.append(line)


        # gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
        #                      low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self,seq_path):
        pass

    def _get_sequence_path(self,seq_id,training_mode="Train"):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):

        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        # Sort the dictionary so it is chronologically ordered

        # Double check that the bounding box width and height are both positive numbers
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        # Check whether the object is visible, if it is, then return true
        visible = torch.ones(bbox.size(dim=0))

        return {'bbox': bbox, 'valid':valid, 'visible':visible}

        #return bbox


    def _get_frame_path(self, seq_path, frame_id):
        # Since the only thing that matters is the id number at the end of the file, we can grab any file, and grab the first part, which is generic
        for names,subdires,files in os.walk(os.path.join(seq_path,"img")):
            generic_name = files[0][:-7]

        # Lets also convert the frame id into a string
        frame_id_str = str(frame_id)

        # If the frame id is a 1 digit number, add 00
        if len(frame_id_str) == 1:
            return os.path.join(seq_path, 'img', f'{generic_name}00{frame_id}.png')

        # If the frame id is a 2 digit number, add 0
        elif len(frame_id_str) == 2:
            return os.path.join(seq_path, 'img', f'{generic_name}0{frame_id}.png')

        # If the frame id is a 3 digit number, add nothing
        elif len(frame_id_str) == 3:
            return os.path.join(seq_path, 'img', f'{generic_name}{frame_id}.png')

        else:
            return ""



    def _get_frame(self, seq_path,frame_id):

        return self.image_loader(self._get_frame_path(seq_path,frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class


    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        #input(seq_path)
        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def convert_cxcywh_2_x1y1wh(self,bbox):

        cx = bbox[0]
        cy = bbox[1]
        w = bbox[2]
        h = bbox[3]

        x1 = np.round(cx - 0.5*w).astype(int)
        y1 = np.round(cy - 0.5*h).astype(int)

        return (x1,y1,w,h)


if '__main__' == __name__:
    root = "/media/atr17/HDD Storage/Datasets_Download/Catheter_Detection/Images/Train"
    seq_path = "/media/atr17/HDD Storage/Datasets_Download/Catheter_Detection/Images/Train/Catheter/Catheter-50"
    dataset = Catheter_tracking(root)
    frame_list, anno_frames, object_meta = dataset.get_frames(seq_id=45, frame_ids=[5])
    print("success!")
    # Frame list gives you the frame arrays
    # Anno_frames
    #print(anno_frames)

    # Something is wrong with the image loader, lets check if it is taking in the images correctly

    #frame_path = dataset._get_frame_path(seq_path, frame_id=1)
    # Apperently everything is functional, the only issue with it is the cv2 loader acting strange, if training fails, try the PIL image loader
