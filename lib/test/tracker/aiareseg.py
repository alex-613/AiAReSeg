import os

import cv2
import torch

from lib.models.aiareseg import build_aiareseg
from lib.test.tracker.utils import Preprocessor, Proprocessor_Seg
from lib.test.tracker.basetracker import BaseTracker
from lib.train.data.processing_utils import sample_target, transform_image_to_crop, image_proc_seg
from lib.utils.box_ops import clip_box
from lib.utils.merge import merge_feature_sequence
from torchvision.ops import masks_to_boxes
import math
from monai.losses import DiceLoss

# For debugging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class AIARESEG(BaseTracker):
    def __init__(self, params, dataset_name):
        super(AIARESEG, self).__init__(params)
        network = build_aiareseg(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.net = network.cuda()
        self.net.eval()
        self.preprocessor = Preprocessor()
        self.proprocessor_seg = Proprocessor_Seg()
        self.state = None
        # For debug
        self.debug = False
        self.frame_id = 0
        # Set the hyper-parameters
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.HYPER, DATASET_NAME):
            self.cache_siz = self.cfg.TEST.HYPER[DATASET_NAME][0]
            self.refer_cap = 1 + self.cfg.TEST.HYPER[DATASET_NAME][1]
            self.threshold = self.cfg.TEST.HYPER[DATASET_NAME][2]
        else:
            self.cache_siz = self.cfg.TEST.HYPER.DEFAULT[0]
            self.refer_cap = 1 + self.cfg.TEST.HYPER.DEFAULT[1]
            self.threshold = self.cfg.TEST.HYPER.DEFAULT[2]
        if self.debug:
            self.save_dir = 'debug'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # For save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.save_all_masks = params.save_all_masks

        self.dice = DiceLoss(reduction='none')
        self.distance_threshold = 10.0

    def initialize(self, image, info: dict, seq_name: str = None,segmentation:bool = False):

        # First perform cropping and generate the masks
        if segmentation == True:
            # The bbox here is the uncropped initial bounding box
            refer_crop, refer_att_mask, refer_seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox = image_proc_seg(image,
                                                                          masks=info['init_mask'],
                                                                          search_area_factor=self.params.search_factor,
                                                                          output_sz=self.params.search_size)


            if data_invalid[0] == True:
                return True

            self.feat_size = self.params.search_size // 16
            refer_img = self.proprocessor_seg.process(refer_crop, refer_att_mask)

            print("sampling complete")

        else:
            # Forward the long-term reference once
            refer_crop, resize_factor, refer_att_mask = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                                      output_sz=self.params.search_size)

            refer_box = transform_image_to_crop(torch.Tensor(info['init_bbox']), torch.Tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.search_size, self.params.search_size]),
                                                normalize=True)

            self.feat_size = self.params.search_size // 16
            refer_img = self.preprocessor.process(refer_crop, refer_att_mask)

            print("sampling complete")


        with torch.no_grad():
            # The reference dictionary contains info about Channel160, channel80, channel40, channel20
            refer_back = self.net.forward_backbone(refer_img)

            refer_dict_list = [refer_back]
            refer_dict = merge_feature_sequence(refer_dict_list)
            refer_mem = self.net.transformer.run_encoder(refer_dict['feat'], refer_dict['mask'], refer_dict['pos'],
                                                         refer_dict['inr'])

        if segmentation == True:

            target_region = torch.nn.functional.interpolate(refer_seg_mask[0], size=(self.feat_size,self.feat_size), mode='bilinear', align_corners=False)
            target_region = target_region.view(self.feat_size * self.feat_size, -1)
            background_region = 1 - target_region
            refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
            embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                                   dim=0).unsqueeze(0)

        else:
            target_region = torch.zeros((self.feat_size, self.feat_size))
            x, y, w, h = (refer_box * self.feat_size).round().int()
            target_region[max(y, 0):min(y + h, self.feat_size), max(x, 0):min(x + w, self.feat_size)] = 1
            target_region = target_region.view(self.feat_size * self.feat_size, -1)
            background_region = 1 - target_region
            refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
            embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                                   dim=0).unsqueeze(0)


        self.refer_mem_cache = [refer_mem]
        self.refer_emb_cache = [torch.bmm(refer_region, embed_bank).transpose(0, 1)]
        self.refer_pos_cache = [refer_dict['inr']]
        self.refer_msk_cache = [refer_dict['mask']]

        #NEW: Adding cache about each of the reference dictonary
        self.refer_temporal_cache = [refer_back]

        self.refer_mem_list = []
        for _ in range(self.refer_cap):
            self.refer_mem_list.append(self.refer_mem_cache[0])
        self.refer_emb_list = []
        for _ in range(self.refer_cap):
            self.refer_emb_list.append(self.refer_emb_cache[0])
        self.refer_pos_list = []
        for _ in range(self.refer_cap):
            self.refer_pos_list.append(self.refer_pos_cache[0])
        self.refer_msk_list = []
        for _ in range(self.refer_cap):
            self.refer_msk_list.append(self.refer_msk_cache[0])
        self.refer_temporal_dict = {}
        for _ in range(self.refer_cap):
            self.refer_temporal_dict[f"{_}"] = self.refer_temporal_cache[0]

        if segmentation == True:
            self.state = info['init_mask']

        else:
            # Save states
            self.state = info['init_bbox']
            if self.save_all_boxes:
                # Save all predicted boxes
                all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
                return {'all_boxes': all_boxes_save}

    def track(self, image, info: dict = None, seq_name: str = None, segmentation: bool = None):
        H, W, _ = image[0].shape
        self.frame_id += 1
        # Get the t-th search region
        if self.frame_id == 1:
            self.init_mask = self.state

        no_mask = True
        iter = 0

        while no_mask == True:
            if segmentation == True:

                if iter == 0:
                    search_crop, search_att_mask, search_seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox = image_proc_seg(image,
                                                                                  masks=self.state,
                                                                                  search_area_factor=self.params.search_factor,
                                                                                  output_sz=self.params.search_size)


                elif (0<iter<20):
                    # If still cant find it, then lets jitter the box a bit and move it in different directions
                    boxes = masks_to_boxes(self.state[0].unsqueeze(0)).squeeze(0).tolist()

                    boxes = self.x1y1x2y2_to_x1y1wh(boxes)
                    # print(f"Multishot, iter {iter}")
                    # print(boxes)
                    if boxes[2] == 0:
                        boxes[2] = 10
                    if boxes[3] == 0:
                        boxes[3] = 10

                    boxes = [torch.tensor(boxes)]

                    enlarged_search_area = self.params.search_factor * (1.0 + (0.01 * iter))
                    # print(f"enlarged_search_area: {enlarged_search_area}")
                    search_crop, search_att_mask, search_seg_mask, data_invalid, resize_factor_W, resize_factor_H, bbox = image_proc_seg(image,
                                                                                  masks=self.state,
                                                                                  jittered_boxes=boxes,
                                                                                  search_area_factor=enlarged_search_area,
                                                                                  output_sz=self.params.search_size)
                    # print(f"Search area, iter {iter}")
                    # print(bbox)
                    if bbox != None:
                        bbox = (bbox[0].tolist(),)

                else:
                    no_mask = False
                    search_crop = (torch.zeros(size=(self.params.search_size,self.params.search_size,3)),)
                    search_att_mask = (torch.zeros(size=(self.params.search_size,self.params.search_size)),)
                    search_seg_mask = (torch.zeros(size=(1,1,self.params.search_size,self.params.search_size)),)
                    data_invalid = (True,)
                    resize_factor_W, resize_factor_H = W/self.params.search_size, H/self.params.search_size
                    bbox = [0,0,0,0]

                if data_invalid[0] == True and iter<=100:
                    iter += 1
                    data_invalid = (True,)
                    continue

                elif data_invalid[0] == True and iter>=100:
                    print("Problem")


                search_img = self.proprocessor_seg.process(search_crop, search_att_mask)

            if segmentation == False:

                search_crop, resize_factor, search_att_mask = sample_target(image, self.state, self.params.search_factor,
                                                                            output_sz=self.params.search_size)  # (x1, y1, w, h)
                search_img = self.preprocessor.process(search_crop, search_att_mask)

            search_dict_list = []
            with torch.no_grad():

                search_back = self.net.forward_backbone(search_img)

                search_back_short = {k: search_back[k] for i,k in enumerate(search_back) if i < 4}
                search_dict_list.append(search_back_short)
                search_dict = merge_feature_sequence(search_dict_list)

                # Run the transformer
                out_embed, search_mem, pos_emb, key_mask = self.net.forward_transformer(search_dic=search_dict,
                                                                                        refer_mem_list=self.refer_mem_list,
                                                                                        refer_emb_list=self.refer_emb_list,
                                                                                        refer_pos_list=self.refer_pos_list,
                                                                                        refer_msk_list=self.refer_msk_list)

                if segmentation==True:
                    out_seg = self.net.forward_segmentation(out_embed, search_outputs=search_back, reference_outputs=self.refer_temporal_dict)

                else:
                    # Forward the corner head
                    out_dict, outputs_coord = self.net.forward_box_head(out_embed)
                    # out_dict: (B, N, C), outputs_coord: (1, B, N, C)

                    pred_iou = self.net.forward_iou_head(out_embed, outputs_coord.unsqueeze(0).unsqueeze(0))


            # Get the final result

            if segmentation == True:

                # Processing 1: Perform thresholding, any mask value < 0.5 is filtered out
                out_mask = out_seg.squeeze(0).squeeze(0)
                out_mask = torch.where(out_mask<0.5,0.0,1.0)
                check_sum = torch.sum(out_mask)  # The check sum ensures that a segmentation mask is actually generated

                if torch.sum(check_sum) == 0:
                    no_mask = True
                    iter += 1
                    continue
                else:
                    no_mask = False
                # Processing 2: Perform mapping, mapping the segmentation mask back into the original image dimensions
                resize_factors = [resize_factor_H, resize_factor_W]

                new_state = [self.map_mask_back(resize_factors=resize_factors, mask=out_mask, bbox=bbox, im_H=H, im_W=W, iter=iter)]

                # Check the intersection
                dice_loss = self.dice(new_state[0], self.state[0])
                IOU = self.dice_loss_to_iou(dice_loss)

                # Please check the bounding box outcome of the new state, if that new outcome generates an empty bounding box, then do not update the state!
                try:
                    state_bbox = masks_to_boxes(new_state[0].unsqueeze(0)).squeeze(0).tolist()
                    state_bbox = self.x1y1x2y2_to_x1y1wh(state_bbox)
                    if (state_bbox[2] != 0) or (state_bbox[3] != 0):
                        self.state = new_state
                        centroid_current = self.calculate_centroid(new_state[0])
                        centroid_old = self.calculate_centroid(self.state[0])
                        euclidean_distance = self.euclidean_distance(centroid_old, centroid_current)
                    else:
                        euclidean_distance = 0
                except:
                    self.state = self.state
                    euclidean_distance = 0


            else:
                pred_boxes = out_dict['pred_boxes'].view(-1, 4)
                # Baseline: Take the mean of all predicted boxes as the final result
                pred_box = (pred_boxes.mean(
                    dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                # Get the final box result
                self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

                predicted_iou = pred_iou['pred_iou'][0][0][0].item()

            # ax.add_patch(plt.Rectangle((self.state[0], self.state[1]), self.state[2], self.state[3], fill=False, color=[0.000, 0.447, 0.741], linewidth=3))
        if data_invalid[0] == True:
            return True

        if segmentation == True:

            if IOU > 0.7:
                # We do not wish to use the IOU as a method of keeping the references, instead,we will store every reference mask possible

                # Then we use all of the reference masks for the next prediction

                # This may be demanding on the computation power of the workstation, but we can reduce the buffer sizes if necessary
                self.feat_size = self.params.search_size // 16

                if len(self.refer_mem_cache) == self.cache_siz:
                    _ = self.refer_mem_cache.pop(1)
                    _ = self.refer_emb_cache.pop(1)
                    _ = self.refer_pos_cache.pop(1)
                    _ = self.refer_msk_cache.pop(1)
                    # New
                    _ = self.refer_temporal_cache.pop(1)

                # The target regions are updated based on previous segementation masks instead
                target_region = torch.nn.functional.interpolate(out_mask.unsqueeze(0).unsqueeze(0), size=(self.feat_size,self.feat_size), mode='bilinear', align_corners=False)
                target_region = target_region.view(self.feat_size * self.feat_size, -1)
                background_region = 1 - target_region
                refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
                embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight], dim=0).unsqueeze(0)
                new_emb = torch.bmm(refer_region, embed_bank).transpose(0, 1)

                self.refer_mem_cache.append(search_mem)
                self.refer_emb_cache.append(new_emb)
                self.refer_pos_cache.append(pos_emb)
                self.refer_msk_cache.append(key_mask)
                # new
                self.refer_temporal_cache.append(search_back)

                self.refer_mem_list = [self.refer_mem_cache[0]]
                self.refer_emb_list = [self.refer_emb_cache[0]]
                self.refer_pos_list = [self.refer_pos_cache[0]]
                self.refer_msk_list = [self.refer_msk_cache[0]]
                # new
                self.refer_temporal_dict = {}
                self.refer_temporal_dict['0'] = self.refer_temporal_cache[0]

                max_idx = len(self.refer_mem_cache) - 1
                ensemble = self.refer_cap - 1

                for part in range(ensemble):
                    temp = max_idx * (part + 1) // ensemble
                    dict_idx = part + 1
                    self.refer_mem_list.append(self.refer_mem_cache[max_idx * (part + 1) // ensemble])
                    self.refer_emb_list.append(self.refer_emb_cache[max_idx * (part + 1) // ensemble])
                    self.refer_pos_list.append(self.refer_pos_cache[max_idx * (part + 1) // ensemble])
                    self.refer_msk_list.append(self.refer_msk_cache[max_idx * (part + 1) // ensemble])
                    self.refer_temporal_dict[f'{dict_idx}'] = self.refer_temporal_cache[max_idx * (part + 1) // ensemble]

        else:
            # Update state
            if predicted_iou > self.threshold:
                if len(self.refer_mem_cache) == self.cache_siz:
                    _ = self.refer_mem_cache.pop(1)
                    _ = self.refer_emb_cache.pop(1)
                    _ = self.refer_pos_cache.pop(1)
                    _ = self.refer_msk_cache.pop(1)
                target_region = torch.zeros((self.feat_size, self.feat_size))
                x, y, w, h = (outputs_coord[0] * self.feat_size).round().int()
                target_region[max(y, 0):min(y + h, self.feat_size), max(x, 0):min(x + w, self.feat_size)] = 1
                target_region = target_region.view(self.feat_size * self.feat_size, -1)
                background_region = 1 - target_region
                refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
                embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                                       dim=0).unsqueeze(0)
                new_emb = torch.bmm(refer_region, embed_bank).transpose(0, 1)
                self.refer_mem_cache.append(search_mem)
                self.refer_emb_cache.append(new_emb)
                self.refer_pos_cache.append(pos_emb)
                self.refer_msk_cache.append(key_mask)

                self.refer_mem_list = [self.refer_mem_cache[0]]
                self.refer_emb_list = [self.refer_emb_cache[0]]
                self.refer_pos_list = [self.refer_pos_cache[0]]
                self.refer_msk_list = [self.refer_msk_cache[0]]
                max_idx = len(self.refer_mem_cache) - 1
                ensemble = self.refer_cap - 1
                for part in range(ensemble):
                    temp = max_idx * (part + 1) // ensemble
                    self.refer_mem_list.append(self.refer_mem_cache[max_idx * (part + 1) // ensemble])
                    self.refer_emb_list.append(self.refer_emb_cache[max_idx * (part + 1) // ensemble])
                    self.refer_pos_list.append(self.refer_pos_cache[max_idx * (part + 1) // ensemble])
                    self.refer_msk_list.append(self.refer_msk_cache[max_idx * (part + 1) // ensemble])

        # For debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 255, 0), thickness=3)
            save_seq_dir = os.path.join(self.save_dir, seq_name)
            if not os.path.exists(save_seq_dir):
                os.makedirs(save_seq_dir)
            save_path = os.path.join(save_seq_dir, '%04d.jpg' % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            # Save all predictions
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N,)
            return {'target_bbox': self.state,
                    'all_boxes': all_boxes_save}

        if self.save_all_masks:
            return {'target_mask': self.state}

        else:
            return {'target_bbox': self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_mask_back(self, mask, resize_factors, bbox, im_H, im_W, iter):

        # This is the bounding box of the previous frame's segmentation mask, not enlarged
        if isinstance(bbox, list) or isinstance(bbox, tuple):
            x1,y1,w,h = bbox[0]

        else:
            x1, y1, x2, y2 = bbox[0][0, ...].tolist()
            w = x2 - x1
            h = y2 - y1
        if iter == 0:
            # You may want to enlarge it in order to make the sizes match
            crop_sz = math.ceil(math.sqrt(w * h) * self.params.search_factor)
        else:
            crop_sz = math.ceil(math.sqrt(w * h) * self.params.search_factor * (1+0.05)*iter)


        x1_new = round(x1 + 0.5 * w - crop_sz * 0.5)
        x2_new = x1_new + crop_sz
        y1_new = round(y1 + 0.5 * h - crop_sz * 0.5)
        y2_new = y1_new + crop_sz

        x1_new_pad = max(0, -x1_new)
        x2_new_pad = max(x2_new - im_W + 1, 0)

        y1_new_pad = max(0, -y1_new)
        y2_new_pad = max(y2_new - im_H + 1, 0)

        padded_W = (x2_new - x2_new_pad) - (x1_new + x1_new_pad)
        padded_H = (y2_new - y2_new_pad) - (y1_new + y1_new_pad)

        # After enlarged, take the mask, shrink it to the size you desire

        mask_orig_cropped = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(padded_H, padded_W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Once shrunk, create a tensor of zeros

        mask_orig = torch.zeros(size=(im_H, im_W))

        # Replace the patch with the mask
        mask_orig[(y1_new+y1_new_pad):(y2_new-y2_new_pad),(x1_new+x1_new_pad):(x2_new-x2_new_pad)] = mask_orig_cropped

        check_sum = torch.sum(mask_orig)

        #ask_resized = torch.nn.interpolate(mask, size=(w,h), mode='bilinear', align_corners=False)

        #mask_original = torch.zeros()

        return mask_orig

    def map_mask_back_new(self, mask, previous_box, resize_factor):

        # Inputs:
        # Mask: The generated mask output
        # Previous box: The bounding box from the last output
        # Resize factor: This tells you what is the original size of the bounding boxes

        H, W = self.state[0].shape
        output_mask = torch.zeros(size=(H, W))

        previous_box = self.x1y1x2y2_to_x1y1wh(previous_box[0].squeeze(0).tolist())

        bbox = masks_to_boxes(mask.unsqueeze(0)) # Remember, this is in the cooridnates of the cropped frame

        x1,y1,x2,y2 = bbox.squeeze(0).tolist()
        w = x2-x1
        h = y2-y1
        cx = x1 + 0.5*w
        cy = y1 + 0.5*h
        bbox = [cx, cy, w, h]
        bbox = [elem/resize_factor[0] for elem in bbox]

        mapped_box = self.map_box_back_new(bbox, resize_factor[0], previous_box)

        # Find the center point
        cx_true = mapped_box[0] + 0.5*mapped_box[2]
        cy_true = mapped_box[1] + 0.5*mapped_box[3]

        x1_true = round(cx_true - 0.5*self.params.search_size/resize_factor[0])
        x2_true = round(x1_true + self.params.search_size/resize_factor[0])

        y1_true = round(cy_true - 0.5*self.params.search_size/resize_factor[0])
        y2_true = round(y1_true + self.params.search_size/resize_factor[0])


        # Find the actual size

        # We will enlarge the mapped box by the w*h*search_factor
        crop_size = math.ceil(mapped_box[2] * mapped_box[3] * self.params.search_factor)
        x1_l = round(mapped_box[0] + 0.5 * mapped_box[2] - crop_size * 0.5)
        x2_l = x1_l + crop_size

        y1_l = round(mapped_box[1] + 0.5 * mapped_box[3] - crop_size * 0.5)
        y2_l = y1_l + crop_size

        resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(round(self.params.search_size/resize_factor[0]), round(self.params.search_size/resize_factor[0])), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        # Replace the image with the actual crop

        try:
            output_mask[y1_true:y2_true, x1_true:x2_true] = resized_mask
        except:
            print(f"Size mismatch error, the mask size is {y2_true-y1_true},{x2_true-x1_true}, and the resized_mask is {round(self.params.search_size/resize_factor[0])}")




        return output_mask

    def map_box_back_new(self, pred_box: list, resize_factor: float, previous_box: list):
        cx_prev, cy_prev = previous_box[0] + 0.5 * previous_box[2], previous_box[1] + 0.5 * previous_box[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor # Half the actual size of the bounding box
        cx_real = cx + (cx_prev - half_side) # The origin of the previous frame, and then add the coordinates of the new frame
        cy_real = cy + (cy_prev - half_side) # So now we are in the coordinate system of the
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def dice_loss_to_iou(self, dice_loss):
        return 1 - dice_loss / (2 - dice_loss)

    def x1y1x2y2_to_x1y1wh(self, bbox):

        x1,y1,x2,y2 = bbox

        w = x2-x1
        h = y2-y1

        return [x1,y1,w,h]

    def _get_jittered_box(self,box,factor_scale, factor_center):
        """
        Jitter the input box.

        Args:
            box: Input bounding box.
            mode: String 'reference' or 'search' indicating reference or search data.

        Returns:
            torch.Tensor: Jittered box.
        """

        noise = torch.exp(torch.randn(2) * factor_scale)
        jittered_size = box[2:4] * noise
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(factor_center).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def calculate_centroid(self, mask):
        total_ones = torch.sum(mask)
        y_moment, x_moment = self.calculate_moments(mask)
        centroid_y = y_moment / total_ones
        centroid_x = x_moment / total_ones
        return centroid_y.item(), centroid_x.item()

    def calculate_moments(self,mask):
        y_moment = torch.sum(torch.arange(mask.shape[0]).float().unsqueeze(1) * mask)
        x_moment = torch.sum(torch.arange(mask.shape[1]).float().unsqueeze(0) * mask)
        return y_moment, x_moment

    def euclidean_distance(self, elem1, elem2):
        output = math.sqrt((elem1[0]-elem2[0])**2 + (elem1[1]-elem2[1])**2)
        return output


def get_tracker_class():
    return AIARESEG
