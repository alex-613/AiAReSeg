import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.ops import masks_to_boxes

import lib.train.data.processing_utils as prutils
from lib.utils import TensorDict

# For debugging
import matplotlib.pyplot as plt
import numpy as np
import cv2

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """
    Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
    through the network. For example, it can be used to crop a search region around the object, apply various data
    augmentations, etc.
    """

    def __init__(self, transform=transforms.ToTensor(), search_transform=None, joint_transform=None):
        """
        Args:
            transform: The set of transformations to be applied on the images.
                       Used only if search_transform is None.
            search_transform: The set of transformations to be applied on the search images.
                              If None, the 'transform' argument is used instead.
            joint_transform: The set of transformations to be applied 'jointly' on the reference and search images.
                             For example, it can be used to convert both reference and search images to grayscale.
        """

        self.transform = {'search': transform if search_transform is None else search_transform,
                          'reference': transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class AIATRACKProcessing(BaseProcessing):
    """
    The processing class used for training LittleBoy. The images are processed in the following way.

    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region)
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        Args:
            search_area_factor: The size of the search region  relative to the target size.
            output_sz: An integer, denoting the size to which the search region is resized.
                       The search region is always square.
            center_jitter_factor: A dict containing the amount of jittering to be applied to the target center before
                                  extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor: A dict containing the amount of jittering to be applied to the target size before
                                 extracting the search region. See _get_jittered_box for how the jittering is done.
            mode: Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames.
        """

        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """
        Jitter the input box.

        Args:
            box: Input bounding box.
            mode: String 'reference' or 'search' indicating reference or search data.

        Returns:
            torch.Tensor: Jittered box.
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """
        Generates proposals by adding noise to the input box.

        Args:
            box: Input box.

        Returns:
            torch.Tensor: Array of shape (num_proposals, 4) containing proposals.
            torch.Tensor: Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box.
                          The IoU is mapped to [-1, 1].
        """

        # Generate proposals
        num_proposals = 16

        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=0.1,
                                                             sigma_factor=[0.03, 0.05, 0.1, 0.2, 0.3])

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def cxcywh_2_x1y1wh(self,box):

        box[0] = box[0] - 0.5* box[2]
        box[1] = box[1] - 0.5* box[3]

        return box

    def __call__(self, data: TensorDict):
        """
        Args:
            data: The input data, should contain the following fields:
                  'reference_images', search_images', 'reference_anno', 'search_anno'

        Returns:
            TensorDict: Output data block with following fields:
                        'reference_images', 'search_images', 'reference_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # If we are using the catheter data, we need to convert the annotation into the catheter format
        #
        # for i in range(len(data['search_anno'])):
        #     data['search_anno'][i] = self.cxcywh_2_x1y1wh(data['search_anno'][i])
        #
        # for i in range(len(data['reference_anno'])):
        #     data['reference_anno'][i] = self.cxcywh_2_x1y1wh(data['reference_anno'][i])


        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'],
                                                                                 bbox=data['search_anno'])
            data['reference_images'], data['reference_anno'] = self.transform['joint'](image=data['reference_images'],
                                                                                       bbox=data['reference_anno'],
                                                                                       new_roll=False)

        for s in ['search', 'reference']:
            # Add a uniform noise to the center pos
            if s in ['reference']:
                jittered_anno = [self._get_jittered_box(data[s + '_anno'][0], 'initial')]
                for a in data[s + '_anno'][1:]:
                    jittered_anno.append(self._get_jittered_box(a, s))
            else:
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print('too small box is found, replace it with new data')
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask = prutils.jittered_center_crop(data[s + '_images'],
                                                                  jittered_anno,
                                                                  data[s + '_anno'],
                                                                  self.search_area_factor[s],
                                                                  self.output_sz[s])

            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, joint=False)

            # Draw the transform out and see how it looks like
            # ax = plt.gca()
            # plt.imshow(data['search_images'][0])
            # ax.add_patch(plt.Rectangle((data['search_anno'][0][0], data['search_anno'][0][1]), data['search_anno'][0][2], data['search_anno'][0][3], fill=False, color = [0.000, 0.447, 0.741], linewidth=3))
            # plt.show()

            if s in ['reference']:
                feat_size = self.output_sz[s] // 16
                data[s + '_region'] = []
                for anno in data[s + '_anno']:
                    target_region = torch.zeros((feat_size, feat_size))
                    x, y, w, h = (anno * feat_size).round().int()
                    target_region[max(y, 0):min(y + h, feat_size), max(x, 0):min(x + w, feat_size)] = 1
                    target_region = target_region.view(feat_size * feat_size, -1)
                    background_region = 1 - target_region
                    data[s + '_region'].append(torch.cat([target_region, background_region], dim=1))

            # Check whether elements in data[s + '_att'] is all 1
            # Which means all of elements are padded
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print('values of original attention mask are all one, replace it with new data')
                    return data
            # More strict conditions: require the down-sampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print('values of down-sampled attention mask are all one, replace it with new data')
                    return data

        # Generate proposals
        iou_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['search_anno']])
        data['search_proposals'] = list(iou_proposals)
        data['proposal_iou'] = list(gt_iou)

        data['valid'] = True

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        # Have a look at what is happening here for each element of the
        return data

class AIATRACKProcessingSeg(BaseProcessing):
    def __init__(self,search_area_factor, center_jitter_factor, scale_jitter_factor, output_sz, mode='pair', settings=None, *args, **kwargs):
        """
        Args:
            search_area_factor: The size of the search region  relative to the target size.
            output_sz: An integer, denoting the size to which the search region is resized.
                       The search region is always square.
            center_jitter_factor: A dict containing the amount of jittering to be applied to the target center before
                                  extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor: A dict containing the amount of jittering to be applied to the target size before
                                 extracting the search region. See _get_jittered_box for how the jittering is done.
            mode: Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames.
        """

        super().__init__(*args, **kwargs)

        self.mode = mode
        self.settings = settings
        self.output_sz = output_sz
        self.search_area_factor = search_area_factor

        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor


    def __call__(self, data: TensorDict):
        """
        Args:
            data: The input data, should contain the following fields:
                  'reference_images', search_images', 'reference_anno', 'search_anno'

        Returns:
            TensorDict: Output data block with following fields:
                        'reference_images', 'search_images', 'reference_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # If we are using the catheter data, we need to convert the annotation into the catheter format


        # temp = data['search_anno'][0]
        # plt_tensor = temp.squeeze(0).repeat(3, 1, 1)
        # plt_tensor = plt_tensor.permute(1, 2, 0).detach().cpu().numpy()
        # plt_tensor = np.where(plt_tensor == 1, [255, 0, 0], 0).astype(float) / 255.0
        #
        # img = data['search_images'][0]
        # output = img.astype(float) / 255.0
        # img = output
        # output = cv2.addWeighted(output, 1, plt_tensor, 0.5, 0)
        #
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(img)
        # axs[1].imshow(output)
        # plt.show()
        # # print('success')


        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'],
                                                                                 mask=data['search_anno'])

            data['reference_images'], data['reference_anno'] = self.transform['joint'](image=data['reference_images'],
                                                                                       mask=data['reference_anno'],
                                                                                       new_roll=False)

        for s in ['search', 'reference']:
            # Add a uniform noise to the center pos
            # if s in ['reference']:
            #     jittered_anno = [self._get_jittered_box(data[s + '_anno'][0], 'initial')]
            #     for a in data[s + '_anno'][1:]:
            #         jittered_anno.append(self._get_jittered_box(a, s))
            # else:
            #     jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            #
            # # Check whether data is valid. Avoid too small bounding boxes
            # w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            # crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            # if (crop_sz < 1).any():
            #     data['valid'] = False
            #     # print('too small box is found, replace it with new data')
            #     return data
            #
            # # TODO: This gets complicated, we need the attention mask but im not sure if we need cropping
            # # Crop image region centered at jittered_anno box and get the attention mask
            # crops, boxes, att_mask = prutils.jittered_center_crop(data[s + '_images'],
            #                                                       jittered_anno,
            #                                                       data[s + '_anno'],
            #                                                       self.search_area_factor[s],
            #                                                       self.output_sz[s])

            bounding_boxes = [self.x1y1x2y2_to_x1y1wh(masks_to_boxes(mask.unsqueeze(0))) for mask in data[s + '_anno']]

            if s in ['reference']:
                jittered_anno = [self._get_jittered_box(bounding_boxes[0], 'initial')]
                for a in bounding_boxes[1:]:
                    jittered_anno.append(self._get_jittered_box(a,s))
            else:
                jittered_anno = [self._get_jittered_box(a.squeeze(0), s) for a in bounding_boxes]

            # Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print('too small box is found, replace it with new data')
                return data


            frames_resized, att_mask, seg_mask, data_invalid,_,_,_ = prutils.image_proc_seg(data[s + '_images'], masks = data[s + '_anno'], jittered_boxes=jittered_anno, search_area_factor=self.search_area_factor[s], output_sz=self.output_sz[s])

            for inva in data_invalid:
                if inva == True:
                    data['valid'] = False
                    return data
            data[s + '_images_o'] = [torch.tensor(ele).permute(2,0,1) for ele in frames_resized]

            # TODO: This will not work without modification
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'] = self.transform[s](
                image=frames_resized, mask=seg_mask, att=att_mask, joint=False)

            # Draw the transform out and see how it looks like
            # ax = plt.gca()
            # plt.imshow(data['search_images'][0])
            # ax.add_patch(plt.Rectangle((data['search_anno'][0][0], data['search_anno'][0][1]), data['search_anno'][0][2], data['search_anno'][0][3], fill=False, color = [0.000, 0.447, 0.741], linewidth=3))
            # plt.show()

            if s in ['reference']:
                feat_size = self.output_sz[s] // 16
                data[s + '_region'] = []
                for anno in data[s + '_anno']:
                    target_region = anno
                    target_region = torch.nn.functional.interpolate(target_region, size=(feat_size,feat_size), mode='bilinear', align_corners=False)
                    target_region = target_region.view(feat_size * feat_size, -1)
                    background_region = 1 - target_region
                    data[s + '_region'].append(torch.cat([target_region, background_region], dim=1))

            # TODO: Double check what is going on here
            # Check whether elements in data[s + '_att'] is all 1
            # Which means all of elements are padded
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print('values of original attention mask are all one, replace it with new data')
                    return data
            # More strict conditions: require the down-sampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print('values of down-sampled attention mask are all one, replace it with new data')
                    return data

        # # Generate proposals, only for detection tasks
        # iou_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['search_anno']])
        # data['search_proposals'] = list(iou_proposals)
        # data['proposal_iou'] = list(gt_iou)

        data['valid'] = True

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        # Have a look at what is happening here for each element of the
        return data

    def generate_bboxes(self,mask):
        # Generating bounding boxes from the segmentation mask for cropping
        non_zero_indices = torch.where(mask)

        min_row = torch.min(non_zero_indices[0])
        min_col = torch.min(non_zero_indices[1])
        max_row = torch.max(non_zero_indices[0])
        max_col = torch.max(non_zero_indices[1])

        bounding_box = torch.tensor([min_row,min_col,max_row,max_col])

        return bounding_box

    def perform_cropping(self, im, seg_mask, target_bb, search_area_factor, output_sz=None):
        # We want to crop the image and the mask

        if not isinstance(target_bb, list):
            x, y, w, h = target_bb.tolist()
        else:
            x, y, w, h = target_bb

        # Crop image
        crop_sz = torch.ceil(torch.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('ERROR: too small bounding box')

        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz

        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # Crop the target mask as well
        im_mask_crop = seg_mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]


        # Pad
        im_crop_padded = torch.nn.functional.pad(im_crop,(y1_pad, y2_pad, x1_pad, x2_pad), value=0)

        # Pad the mask
        mask_crop_padded = torch.nn.functional.pad(im_mask_crop, (y1_pad, y2_pad, x1_pad, x2_pad), value=0)


        # TODO:
        # Deal with attention mask
        H, W, _ = im_crop_padded.shape
        att_mask = np.ones((H, W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0

        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            im_crop_padded = torch.nn.functional.interpolate(im_crop_padded, size=(output_sz, output_sz), mode='bilinear',
                                            align_corners=False)
            mask_crop_padded = torch.nn.functional.interpolate(mask_crop_padded, size=(output_sz, output_sz), mode='bilinear',
                                            align_corners=False)
            att_mask = torch.nn.functional.interpolate(att_mask, size=(output_sz, output_sz), mode='bilinear',
                                            align_corners=False)
            return im_crop_padded, mask_crop_padded, resize_factor, att_mask
        else:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0

    def _get_jittered_box(self, box, mode):
        """
        Jitter the input box.

        Args:
            box: Input bounding box.
            mode: String 'reference' or 'search' indicating reference or search data.

        Returns:
            torch.Tensor: Jittered box.
        """

        noise = torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        jittered_size = box[2:4] * noise
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def x1y1x2y2_to_x1y1wh(self,bbox):

        # Convert to the standard format

        x1,y1,x2,y2 = bbox[0,...].tolist()
        w = x2-x1
        h = y2-y1

        return torch.tensor([x1,y1,w,h])
