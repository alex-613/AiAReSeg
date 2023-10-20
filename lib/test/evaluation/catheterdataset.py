import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import torch


class CatheterDataset(BaseDataset):
    """
    The catheter tracking dataset consists of 50 training sequences and an additional 15 testing sequences.

    All of them are simulated using the ray tracing algorithm from Imfusion Inc.

    """

    # Constructor

    def __init__(self, subset='Val'):
        super().__init__()
        self.base_path = self.env_settings.catheter_path
        print("Catheter Base Path:")
        print(self.base_path)
        self.base_path = os.path.join(self.base_path, subset)

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

    def _get_sequence_list(self,subset):
        if subset == 'Val':
            # We grab all of the sequences in the folder by doing a walk
            seq_list = []
            for names, subdires, files in os.walk(os.path.join(self.base_path, "Catheter")):

                # Now we can loop through the directory in order to extract the sequences
                for subdir in subdires:
                    if subdir != 'img':
                        seq_list.append(subdir)

            return sorted(seq_list, key=lambda x:x[-2:])

    def get_sequence_list(self, subset='Val'):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])


    def _read_bb_anno(self, seq_path):
        # For each of the folders of sequences, you need to generate the annotations and put them into an annotation file
        seq = seq_path.split('/')[-1]
        bb_anno_file = os.path.join(seq_path, f'gt_{seq}.txt')
        gt = []

        with open(bb_anno_file) as f:
            for line in f.readlines():
                line = line[2:-3]
                line = line.split(",")
                line = [float(i) for i in line]
                print(line)
                gt.append(line)


        return np.array(gt)

    def _get_sequence_path(self,seq_id,training_mode="Val"):
        seq_name = self.sequence_list[seq_id-922]
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
        anno_path = os.path.join(self.base_path, self.base_path, class_name, sequence_name, f'gt_{sequence_name}.txt')

        seq_path = os.path.join(self.base_path, class_name, sequence_name)

        frames_path = os.path.join(self.base_path, class_name, sequence_name, 'img')

        ground_truth_rect = self._read_bb_anno(seq_path)

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
        return Sequence(sequence_name, frames_list, 'catheter', ground_truth_rect.reshape(-1, 4), object_class=target_class, target_visible=full_occlusion)
        # return Sequence(sequence_name, frames_list, 'catheter', ground_truth_rect,
        #                 object_class=target_class, target_visible=full_occlusion)

    def __len__(self):
        return len(self.sequence_list)


if "__main__" == __name__:

    dataset = CatheterDataset(subset='Val')
    sequence = dataset._construct_sequence(sequence_name='Catheter-690')
    seq_len = dataset._get_seq_len(seq_id=690)
    print(seq_len)


