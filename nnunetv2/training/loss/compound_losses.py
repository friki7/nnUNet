import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
        

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        """
        Compute NT-Xent Contrastive Loss
        - feature_vectors: Tensor of shape (batch_size, num_features)
        - labels: Binary labels (1 for airway, 0 for non-airway)
        """
        device = feature_vectors.device
        batch_size = feature_vectors.shape[0]

        # Normalize feature vectors
        feature_vectors = F.normalize(feature_vectors, p=2, dim=1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(feature_vectors, feature_vectors.T) / self.temperature

        # Ensure labels are properly shaped
        labels = labels.view(batch_size, 1)
        label_matrix = torch.eq(labels, labels.T).float().to(device)

        # Compute contrastive loss
        loss = F.cross_entropy(similarity_matrix, label_matrix)
        return loss


class DC_CE_Contrastive_Loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, contrastive_temp=0.1, weight_ce=1, weight_dice=1, weight_contrastive=0.5, ignore_label=None):
        """
        Combines Dice Loss + Cross-Entropy Loss + Contrastive Loss

        :param soft_dice_kwargs: Dice Loss settings
        :param ce_kwargs: Cross-Entropy settings
        :param contrastive_temp: Temperature parameter for Contrastive Loss
        :param weight_ce: Weight of Cross-Entropy Loss
        :param weight_dice: Weight of Dice Loss
        :param weight_contrastive: Weight of Contrastive Loss
        :param ignore_label: Label to ignore in segmentation
        """
        super(DC_CE_Contrastive_Loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_contrastive = weight_contrastive
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.contrastive = ContrastiveLoss(temperature=contrastive_temp)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, feature_vectors: torch.Tensor):
        """
        Compute total loss: Dice Loss + Cross-Entropy Loss + Contrastive Loss

        :param net_output: Segmentation output (B, C, X, Y, Z)
        :param target: Ground truth labels (B, C, X, Y, Z)
        :param feature_vectors: Extracted encoder features (B, num_features)
        """
        # Ensure target is correctly shaped
        if target.ndim == 5:  # If target has batch & spatial dims
            target_dice = target
            target_ce = target[:, 0]
        else:
            target_dice = target.unsqueeze(1)  # Add a channel dimension if missing
            target_ce = target

        # Compute Dice and Cross-Entropy Loss
        dc_loss = self.dc(net_output, target_dice) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target_ce) if self.weight_ce != 0 else 0

        # Normalize feature vectors before contrastive loss
        feature_vectors = F.normalize(feature_vectors, p=2, dim=1)

        # Compute Contrastive Loss on feature vectors
        contrastive_loss = self.contrastive(feature_vectors, target_ce) if self.weight_contrastive != 0 else 0

        # Compute total loss
        total_loss = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_contrastive * contrastive_loss

        return total_loss

