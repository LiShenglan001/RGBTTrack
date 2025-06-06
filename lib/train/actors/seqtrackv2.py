from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.train.admin import multigpu
import torch.nn as nn
from lib.utils.misc import NestedTensor
import torch.nn.functional as F


class SeqTrackV2Actor(BaseActor):
    """ Actor for training the SeqTrackv2"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        self.bins = net.bins
        self.instruct_tokens = net.decoder.instruct_tokens
        self.seq_format = net.seq_format

    def __call__(self, data, current_epoch=None):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        outputs, target_seqs, volume_loss = self.forward_pass(data, current_epoch=current_epoch)

        # compute losses
        loss, status = self.compute_losses(outputs, target_seqs, volume_loss)

        return loss, status

    def forward_pass(self, data, current_epoch=None):
        n, b, _, _, _ = data['search_images'].shape   # n,b,c,h,w
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(b,dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b,dim=0)

        bins = self.bins # coorinate token
        instruct_tokens = self.instruct_tokens

        # box of search region
        targets = data['search_anno'].permute(1,0,2).reshape(-1, data['search_anno'].shape[2])   # x0y0wh
        targets = box_xywh_to_xyxy(targets)   # x0y0wh --> x0y0x1y1
        targets = torch.max(targets, torch.tensor([0.]).to(targets)) # Truncate out-of-range values
        targets = torch.min(targets, torch.tensor([1.]).to(targets))

        # box of template region
        z_anno = data['template_anno'].permute(1, 0, 2)
        x_anno = data['search_anno'].permute(1, 0, 2)
        # z_anno = data['template_anno'].permute(1, 0, 2).reshape(-1, data['template_anno'].shape[2])  # x0y0wh
        # z_anno = box_xywh_to_xyxy(z_anno)  # x0y0wh --> x0y0x1y1
        # z_anno = torch.max(z_anno, torch.tensor([0.]).to(z_anno))  # Truncate out-of-range values
        # z_anno = torch.min(z_anno, torch.tensor([1.]).to(z_anno))

        # different formats of sequence, for ablation study
        seq_format = self.seq_format
        if seq_format != 'corner':
            targets = box_xyxy_to_cxcywh(targets)

        box = (targets * (bins - 1)).int() # discretize the coordinates

        if seq_format == 'whxy':
            box = box[:, [2, 3, 0, 1]]

        batch = box.shape[0]

        # input sequence
        instruct_tokens_batch = [instruct_tokens[key] for key in data['dataset']]
        instruct_tokens_batch = torch.tensor(instruct_tokens_batch).to(box).view(-1,1)

        input_seqs = torch.cat([instruct_tokens_batch, box], dim=1)
        input_seqs = input_seqs.reshape(b, n, input_seqs.shape[-1])
        input_seqs = input_seqs.flatten(1)

        # target sequence
        target_end = torch.ones([batch, 1]).to(box) * instruct_tokens['end']
        target_seqs = torch.cat([box, target_end], dim=1)
        target_seqs = target_seqs.reshape(b, n, target_seqs.shape[-1])
        target_seqs = target_seqs.flatten()
        target_seqs = target_seqs.type(dtype=torch.int64)

        # text input
        text_data = NestedTensor(data['nl_token_ids'].permute(1, 0), data['nl_token_masks'].permute(1,0))

        text_src = self.net(text_data=text_data, mode='language')
        feature_xz, volume_loss = self.net(template_list=template_list, search_list=search_list,
                              text_src=text_src,
                              seq=input_seqs, z_anno=z_anno, x_anno=x_anno, current_epoch=current_epoch, mode='encoder') # forward the encoder
        outputs = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")

        outputs = outputs[-1].view(-1, outputs.size(-1))

        return outputs, target_seqs, volume_loss

    def compute_losses(self, outputs, targets_seq, volume_loss, return_status=True):
        # Get loss
        ce_loss = self.objective['ce'](outputs, targets_seq)
        # x_gt_mask = self.create_mask(targets)
        # x_gt_mask = x_gt_mask.unfold(1, 16, 16).unfold(2, 16, 16)  # to match the patch embedding
        # x_gt_mask = x_gt_mask.mean(dim=(3, 4)).flatten(1)
        # hard_label = (x_gt_mask > 0.5).long()
        # gamma = 2.0
        # alpha = 0.9
        # decision = decision.permute(0, 2, 1)
        # pt = F.softmax(decision, dim=1)
        # log_pt = F.log_softmax(decision, dim=1)
        # focal_weight = (1-pt).pow(2)
        # focal_loss = - (focal_weight * log_pt).gather(1, hard_label.unsqueeze(1)).squeeze(1).mean()
        # decision = decision.detach().cpu().numpy()
        # class_loss = F.cross_entropy(decision.permute(0, 2, 1), hard_label, weight=torch.tensor([1.0, 22.0]).cuda())
        # weighted sum
        loss = self.loss_weight['ce'] * ce_loss + 0.2 * volume_loss

        outputs = outputs.softmax(-1)
        outputs = outputs[:, :self.bins]
        value, extra_seq = outputs.topk(dim=-1, k=1)
        boxes_pred = extra_seq.squeeze(-1).reshape(-1,5)[:, 0:-1]
        boxes_target = targets_seq.reshape(-1,5)[:,0:-1]
        boxes_pred = box_cxcywh_to_xyxy(boxes_pred)
        boxes_target = box_cxcywh_to_xyxy(boxes_target)
        iou = box_iou(boxes_pred, boxes_target)[0].mean()

        if return_status:
            # status for log
            status = {"Loss/total": loss.item(),
                      "IoU": iou.item(),
                      "volume_loss": volume_loss.item()}
            return loss, status
        else:
            return loss

    def create_mask(self, image_anno):
        height = 256
        width = 256

        x0 = ((image_anno[:, 0]-image_anno[:, 2]/2) * width).unsqueeze(1)
        y0 = ((image_anno[:, 1]-image_anno[:, 3]/2) * height).unsqueeze(1)
        w = (image_anno[:, 2] * width).unsqueeze(1)
        h = (image_anno[:, 3] * height).unsqueeze(1)

        # Generate pixel indices
        x_indices = torch.arange(width, device=image_anno.device)
        y_indices = torch.arange(height, device=image_anno.device)

        # Create masks for x and y coordinates within the bounding boxes
        x_mask = ((x_indices >= x0) & (x_indices < x0 + w)).float()
        y_mask = ((y_indices >= y0) & (y_indices < y0 + h)).float()

        # Combine x and y masks to get final mask
        mask = x_mask.unsqueeze(1) * y_mask.unsqueeze(2) # (b,h,w)
        return mask

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.objective['ce'].to(device)