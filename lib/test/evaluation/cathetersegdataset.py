import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import torch
import cv2


class CatheterSegDataset(BaseDataset):
    """
    The catheter tracking dataset consists of 50 training sequences and an additional 15 testing sequences.

    All of them are simulated using the ray tracing algorithm from Imfusion Inc.

    """

    # Constructor

    def __init__(self, subset='Val'):
        super().__init__()
        self.base_path = self.env_settings.catheterseg_path
        print("Catheter Base Path:")
        print(self.base_path)
        self.base_path = os.path.join(self.base_path, 'Images', subset)
        self.data_path = self.env_settings.catheterseg_path

        self.sequence_list = self._get_sequence_list(subset)

        self.clean_list = self.clean_seq_list()

    # A clean sequence list method that grabs the class of each sequence, in our case there is only one class

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)

        return clean_lst

    # A get sequence list method at runs the construct sequence method

    def sort_seq_names(self,name):
        parts = name.split("-")
        return int(parts[1])

    def _get_sequence_list(self,subset):
        if subset == 'Val':
            # We grab all of the sequences in the folder by doing a walk
            seq_list = []
            for names, subdires, files in os.walk(os.path.join(self.base_path, "Catheter")):

                # Now we can loop through the directory in order to extract the sequences
                for subdir in subdires:
                    if subdir != 'img':
                        seq_list.append(subdir)

            return sorted(seq_list, key=self.sort_seq_names)

    def get_sequence_list(self, subset='Val'):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    # def _read_bb_anno(self, seq_path):
    #     # For each of the folders of sequences, you need to generate the annotations and put them into an annotation file
    #     seq = seq_path.split('/')[-1]
    #     bb_anno_file = os.path.join(seq_path, f'gt_{seq}.txt')
    #     gt = []
    #
    #     with open(bb_anno_file) as f:
    #         for line in f.readlines():
    #             line = line[2:-3]
    #             line = line.split(",")
    #             line = [float(i) for i in line]
    #             print(line)
    #             gt.append(line)
    #
    #
    #     return np.array(gt)

    def _read_mask_anno(self,seq_path):
        # Here is the segmentation masks, load them and then put it into the same tensor
        # This may be too big, if it is then try to reduce the number of images used
        seq = seq_path.split('/')[-1]
        mask_path = os.path.join(self.data_path,'Masks','Val',seq) # This put you into the mask folder for the sequence you are looking at

        # Now we start to load the segmentation masks
        gt = []
        filenames = os.listdir(mask_path)
        filenames = sorted(filenames)
        # filenames_index = int(np.floor(0.9*len(filenames)))
        # filenames = filenames[filenames_index:]
        for filename in filenames:
            if filename.endswith(".png"):
                path = os.path.join(mask_path, filename)
                mask = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                mask_tensor = torch.tensor(mask)
                # Convert the image such that we will only have the label 1 for the catheter, or else it makes it zero
                mask_tensor = torch.where(mask_tensor==2, 1, 0).float()
                sum_check = mask_tensor.sum()
                #mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0),(320,320))
                gt.append(mask_tensor)

        return torch.stack(gt)


    def _get_sequence_path(self,seq_id,training_mode="Val"):
        seq_name = self.sequence_list[seq_id-716]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(self.base_path,class_name, class_name + '-' + vid_id)

    # A construct sequence method that: Grabs the class name, the ground truth annotations, whether the object is occluded, whether the object is out of view, whether the target is visible, and finally arranges the frames list in the form of a sequence.

    def _get_seq_len(self, seq_id):
        path = self._get_sequence_path(seq_id,training_mode="Val")

        #print(path)
        for seq_paths, seq_subdirs, seq_names in os.walk(os.path.join(path)):
            #print(seq_names)
            seq_names = seq_names

        # Now we can grab the length of that list
        return len(seq_names)

    def _construct_sequence(self,sequence_name):

        # This gives you the class name, which for now is just catheter
        class_name = sequence_name.split('-')[0]

        # This will give you the sequence number, which will range from 50-65
        sequence_number = int(sequence_name.split('-')[1])

        # Join the base path, with the class, with the sequence name, and then finally the ground truth file name
        anno_path = os.path.join(self.base_path, class_name, sequence_name, f'gt_{sequence_name}.txt')

        seq_path = os.path.join(self.base_path, class_name, sequence_name)

        #frames_path = os.path.join(self.base_path, class_name, sequence_name, 'img')
        frames_path = os.path.join(self.base_path, class_name, sequence_name)

        #ground_truth_rect = self._read_bb_anno(seq_path)
        ground_truth_mask = self._read_mask_anno(seq_path)

        #ground_truth_rect = None
        seq_len = self._get_seq_len(sequence_number)


        # The number of zeros will depend on the number of frames in the folder

        full_occlusion = np.zeros(seq_len)

        out_of_view = np.zeros(seq_len)

        # We now want to grab the path of each of the frames

        # Easiest way is to do a walk across the files again

        frames_list = []
        for names, subdires, files in os.walk(os.path.join(frames_path)):
            # print(names)
            # print(subdires)
            # print(files)
            for file in files:

                frames_list.append(os.path.join(frames_path, file))

        frames_list = sorted(frames_list)
        target_class = class_name

        #return Sequence(sequence_name, frames_list, 'catheter_tracking', ground_truth_rect.view(-1, 4), object_class=target_class, target_visible=full_occlusion)
        return Sequence(sequence_name, frames_list, 'catheter', None, ground_truth_seg=ground_truth_mask, object_class=target_class, target_visible=full_occlusion)
        # return Sequence(sequence_name, frames_list, 'catheter', ground_truth_rect,
        #                 object_class=target_class, target_visible=full_occlusion)

    def __len__(self):
        return len(self.sequence_list)


# if "__main__" == __name__:
#
#     dataset = CatheterDataset(subset='Val')
#     sequence = dataset._construct_sequence(sequence_name='Catheter-690')
#     seq_len = dataset._get_seq_len(seq_id=690)
#     print(seq_len)


