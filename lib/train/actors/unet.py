from monai.networks.nets import UNet
from . import BaseActor
import torch
from lib.utils.misc import NestedTensor
from lib.utils.merge import merge_feature_sequence

class UNetActor(BaseActor):
    def __init__(self, net, objective, settings):
        super().__init__(net, objective)
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
        # TODO: For each of these lines here, please implement a new version for segmentation case
        out_dict = self.forward_pass(data) # Done (added torch.no_grad())

        # Compute losses
        # Process the groundtruth
        if self.settings.segmentation:
            gt_anno = data['search_anno'].squeeze(2).squeeze(2).permute(1,0,2,3)
            loss, status = self.compute_losses_seg(out_dict, gt_anno, return_status=True)
            return loss, status

    def forward_pass(self, data):

        # We would need to generalize this into the segmentation domain as well!
        # This would mean redesigning the architecture to take into account the

        # Process the search regions (t-th frame)

        # Assume data is good to go into the network straight away

        output = self.net(data)



