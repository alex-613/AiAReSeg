import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_feature_sequence
from lib.utils.misc import NestedTensor
from . import BaseActor
import numpy as np
import cv2

# For debugging:
import matplotlib.pyplot as plt


class AIARESEGActor(BaseActor):
    """
    Actor for training.
    """

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # Batch size

    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'reference', 'search', 'gt_bbox'.
            reference_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)

        Returns:
            loss: The training loss.
            status: Dict containing detailed losses.
        """

        # Forward pass
        out_dict = self.forward_pass(data)

        # Compute losses
        # Process the groundtruth
        if self.settings.segmentation:
            gt_anno = data['search_anno'].squeeze(2).squeeze(2).permute(1,0,2,3)
            loss, status = self.compute_losses_seg(out_dict, gt_anno, return_status=True)
            return loss, status

        else:
            gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)
            loss, status = self.compute_losses(out_dict, gt_bboxes[0], data['proposal_iou'])
            return loss, status



    def forward_pass(self, data):

        # We would need to generalize this into the segmentation domain as well!
        # This would mean redesigning the architecture to take into account the

        # Process the search regions (t-th frame)
        search_dict_list = []
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        # # Denorm
        # search_img_copy = search_img.permute(0,2,3,1).detach().cpu() #(b,320,320,3)
        # mean = torch.tensor([0.485, 0.465, 0.406])
        # std = torch.tensor([0.229, 0.224, 0.225])
        # search_img_denorm = (search_img_copy * std) + mean
        #
        # # Make it plottable
        # search_img_plot = search_img_denorm[0,...].numpy()


        # plt.imshow(search_img_plot)
        # plt.title('Cropped search image, denormalized')
        # plt.show()


        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        if self.settings.segmentation: # If doing segmentation then freeze the weights
            with torch.no_grad():
                search_back = self.net(img=NestedTensor(search_img, search_att), mode='backbone')
                # Grab the first 4 elements, these are needed for transformer
                # The latter 4 elements are needed for the segmentation decoder
                search_back_short = {k: search_back[k] for i,k in enumerate(search_back) if i < 4}
                search_dict_list.append(search_back_short)
                search_dict = merge_feature_sequence(search_dict_list)

                # #########Plotting attention maps for debugging##########
                # # We plot the original, and then all of the attention maps in subsequent layers
                # copy_src = search_dict['feat']
                # copy_src = copy_src.permute(1, 2, 0).view(-1, 256, 20, 20)
                #
                # plot_img = copy_src[0, 0, ...].detach().cpu().numpy().astype(float)
                # min_val = np.min(plot_img)
                # max_val = np.max(plot_img)
                #
                # new_min = 0.0
                # new_max = 1.0
                #
                # plot_img_transformed = new_min + ((plot_img - min_val) * (new_max - new_min)) / (max_val - min_val)
                #
                # plt.imshow(plot_img_transformed)
                # plt.title('Plotting the attention maps')
                # plt.show()
                # #########Plotting for debugging##########

                # Process the reference frames
                feat_dict_list = []
                refer_reg_list = []
                temporal_dict = {}
                for i in range(data['reference_images'].shape[0]):
                    reference_dict_list = []
                    reference_img_i = data['reference_images'][i].view(-1, *data['reference_images'].shape[
                                                                            2:])  # (batch, 3, 320, 320)
                    reference_att_i = data['reference_att'][i].view(-1, *data['reference_att'].shape[2:])  # (batch, 320, 320)
                    # temp = NestedTensor(reference_img_i, reference_att_i)
                    # input(temp.mask.shape)
                    output = self.net(img=NestedTensor(reference_img_i, reference_att_i), mode='backbone')
                    temporal_dict[f'{i}'] = output
                    ref_back_short = {k: output[k] for i, k in enumerate(output) if i < 4}
                    reference_dict_list.append(ref_back_short)
                    feat_dict_list.append(merge_feature_sequence(reference_dict_list))
                    refer_reg_list.append(data['reference_region'][i])

                # Run the transformer and compute losses
                out_embed, _, _, _ = self.net(search_dic=search_dict, refer_dic_list=feat_dict_list,
                                              refer_reg_list=refer_reg_list, mode='transformer')
        else:
            search_back = self.net(img=NestedTensor(search_img, search_att), mode='backbone')
            search_back_short = {k: search_back[k] for i, k in enumerate(search_back) if i < 4}
            search_dict_list.append(search_back_short)
            search_dict = merge_feature_sequence(search_dict_list)

            # Process the reference frames
            feat_dict_list = []
            refer_reg_list = []
            for i in range(data['reference_images'].shape[0]):
                reference_dict_list = []
                reference_img_i = data['reference_images'][i].view(-1, *data['reference_images'].shape[
                                                                        2:])  # (batch, 3, 320, 320)
                reference_att_i = data['reference_att'][i].view(-1,
                                                                *data['reference_att'].shape[2:])  # (batch, 320, 320)
                # temp = NestedTensor(reference_img_i, reference_att_i)
                # input(temp.mask.shape)
                output = self.net(img=NestedTensor(reference_img_i, reference_att_i), mode='backbone')

                reference_dict_list.append(output)
                feat_dict_list.append(merge_feature_sequence(reference_dict_list))
                refer_reg_list.append(data['reference_region'][i])

            # Run the transformer and compute losses
            out_embed, _, _, _ = self.net(search_dic=search_dict, refer_dic_list=feat_dict_list,
                                          refer_reg_list=refer_reg_list, mode='transformer')


        if self.settings.segmentation:

            #out_seg = self.net(out_embed=out_embed, mode='segmentation',seg20=search_back['Channel20'], seg40=search_back['Channel40'], seg80=search_back['Channel80'], seg160=search_back['Channel160'], pos_emb=search_dict['pos'])
            out_seg = self.net(out_embed=out_embed, mode='segmentation', search_outputs=search_back, reference_outputs=temporal_dict)
            # threshold = 0.5
            # out_seg = out_seg > threshold
            # out_seg = out_seg.float()
            # out_seg = torch.tensor(out_seg,requires_grad=True)
            #
            # B,_, _, _ = out_seg.shape
            #
            # fig, axs = plt.subplots(3, B)
            #
            # for sample in range(B):
            #
            #     debug_mask = out_seg[sample, ...].squeeze(0).detach().cpu().numpy()
            #     debug_img = data['search_images_o'][0,sample,...].detach().cpu().permute(1,2,0).numpy()
            #     gt_mask = data['search_anno'][0,sample,...].squeeze(0).detach().cpu().permute(1,2,0).repeat(1,1,3)
            #     gt_mask = torch.where(gt_mask==1, torch.tensor([0.,1.,0.]), torch.tensor([0.,0.,0.])).numpy().astype(float)
            #
            #
            #     mask_normalized = debug_mask.astype(float)
            #     mask_normalized = np.where(mask_normalized>0.5, 1.0, 0.0)
            #     mask_normalized = np.repeat(mask_normalized[:,:,np.newaxis],3,axis=2)
            #     mask_normalized = np.where(mask_normalized==1., [1.,0.,0.],[0.,0.,0.])
            #
            #     img_normalized = debug_img.astype(float)/255.0
            #
            #     mixed_output = cv2.addWeighted(img_normalized, 1, mask_normalized, 0.5, 0)
            #     mixed_output = cv2.addWeighted(mixed_output, 1, gt_mask, 0.5, 0)
            #
            #     axs[0, sample].imshow(mask_normalized)
            #     axs[0, sample].title.set_text(f'Model Mask frame {sample}')
            #     axs[1, sample].imshow(mixed_output)
            #     axs[1, sample].title.set_text(f'Image frame {sample}')
            #     axs[2, sample].imshow(gt_mask)
            #     axs[2, sample].title.set_text(f'Gt frame {sample}')
            #
            # plt.show()


            return out_seg

        # Forward the corner head
        else:
            out_dict = self.net(out_embed=out_embed, proposals=data['search_proposals'],
                                mode='heads')  # out_dict: (B, N, C), outputs_coord: (1, B, N, C)

            return out_dict

    def compute_losses(self, pred_dict, gt_bbox, iou_gt, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError('ERROR: network outputs is NaN! stop training')
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # Compute GIoU and IoU
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # Compute L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        iou_pred = pred_dict['pred_iou']
        iou_loss = self.objective['iou'](iou_pred, iou_gt)


        # Weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'iou'] * iou_loss
        if return_status:
            # Status for log
            mean_iou = iou.detach().mean()
            status = {'Ls/total': loss.item(),
                      'Ls/giou': giou_loss.item(),
                      'Ls/l1': l1_loss.item(),
                      'Ls/iou': iou_loss.item(),
                      'IoU': mean_iou.item()}
            return loss, status
        else:
            return loss

    def compute_losses_seg(self,out_seg, gt_mask, return_status=True):

        bce_loss = self.objective['BCE'](out_seg,gt_mask)
        try:
            IOU_loss = self.objective['mask_iou'](out_seg, gt_mask)
        except:
            IOU_loss = torch.tensor(0.0).cuda()

        mse = self.objective['MSE'](out_seg,gt_mask)

        # Weighted sum
        loss = self.loss_weight['BCE'] * bce_loss + self.loss_weight['mask_iou'] * IOU_loss + self.loss_weight['MSE'] * mse

        if return_status:
            # Status for log

            status = {'Ls/total': loss.item(),
                      'Ls/bce': bce_loss.item(),
                      'Ls/iou': IOU_loss.item(),
                      'Ls/mse': mse.item()}
            return loss, status
        else:
            return loss

    def find_centroid(self, mask):
        """Grab a mask, and then generate the centroid of that mask, the mask will have a dimension of (8, 1, 320, 320)"""
        rows, cols = torch.where(mask > 0)
        num_pixels = len(rows)

        centroid_x = torch.sum(cols).float() / num_pixels
        centroid_y = torch.sum(rows).float() / num_pixels

        return centroid_x, centroid_y

    def convert_cxcywh_2_x1y1wh(self,bbox):

        cx = bbox[0]
        cy = bbox[1]
        w = bbox[2]
        h = bbox[3]

        x1 = torch.round(cx - 0.5*w).int()
        y1 = torch.round(cy - 0.5*h).int()

        return (x1,y1,w,h)

