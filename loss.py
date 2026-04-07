import os
import time
import glob
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv


## for reconstruction [only recon loss on miss part]
class MaskedReconLoss(nn.Module):

    def __init__(self):
        super(MaskedReconLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, recon_input, target_input, input_mask, umask, adim, tdim, vdim):
        """ ? => refer to spk and modality
        recon_input  -> ? * [seqlen, batch, dim]
        target_input -> ? * [seqlen, batch, dim]
        input_mask   -> ? * [seqlen, batch, dim]
        umask        -> [batch, seqlen]
        """
        #assert len(recon_input) == 1
        recon = recon_input[0] # [seqlen, batch, dim]
        target = target_input[0] # [seqlen, batch, dim]
        mask = input_mask[0] # [seqlen, batch, 3]

        recon  = torch.reshape(recon, (-1, recon.size(2)))   # [seqlen*batch, dim]
        target = torch.reshape(target, (-1, target.size(2))) # [seqlen*batch, dim]
        mask   = torch.reshape(mask, (-1, mask.size(2)))     # [seqlen*batch, 3] 1(exist); 0(mask)
        umask = torch.reshape(umask, (-1, 1)) # [seqlen*batch, 1]

        A_rec = recon[:, :adim]
        L_rec = recon[:, adim:adim+tdim]
        V_rec = recon[:, adim+tdim:]
        A_full = target[:, :adim]
        L_full = target[:, adim:adim+tdim]
        V_full = target[:, adim+tdim:]
        A_miss_index = torch.reshape(mask[:, 0], (-1, 1))
        L_miss_index = torch.reshape(mask[:, 1], (-1, 1))
        V_miss_index = torch.reshape(mask[:, 2], (-1, 1))

        loss_recon1 = self.loss(A_rec*umask, A_full*umask) * -1 * (A_miss_index - 1)
        loss_recon2 = self.loss(L_rec*umask, L_full*umask) * -1 * (L_miss_index - 1)
        loss_recon3 = self.loss(V_rec*umask, V_full*umask) * -1 * (V_miss_index - 1)
        loss_recon1 = torch.sum(loss_recon1) / adim
        loss_recon2 = torch.sum(loss_recon2) / tdim
        loss_recon3 = torch.sum(loss_recon3) / vdim
        loss_recon = (loss_recon1 + loss_recon2 + loss_recon3) / torch.sum(umask)

        return loss_recon


## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):

    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len, n_classes]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1,1) # [batch*seq_len, 1]
        target = target.view(-1,1) # [batch*seq_len, 1]
        pred = F.log_softmax(pred, 1) # [batch*seqlen, n_classes]
        loss = self.loss(pred*umask, (target*umask).squeeze().long()) / torch.sum(umask) 
        return loss


## for cmumosi and cmumosei loss calculation
class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch*seq_len]
        """
        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]
        umask = umask.view(-1, 1) # [batch*seq_len, 1]
        loss = self.loss(pred*umask, target*umask) / torch.sum(umask)
        return loss


## MAE loss for cmumosi and cmumosei (more robust to outliers)
class MaskedMAELoss(nn.Module):

    def __init__(self):
        super(MaskedMAELoss, self).__init__()
        self.loss = nn.L1Loss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch*seq_len]
        """
        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]
        umask = umask.view(-1, 1) # [batch*seq_len, 1]
        loss = self.loss(pred*umask, target*umask) / torch.sum(umask)
        return loss


## Huber loss for cmumosi and cmumosei (combines MSE and MAE advantages)
## - Behaves like MSE for small errors (smooth gradients)
## - Behaves like MAE for large errors (robust to outliers)
class MaskedHuberLoss(nn.Module):

    def __init__(self, delta=1.0):
        """
        Args:
            delta: threshold where loss transitions from quadratic to linear
                   smaller delta -> more like MAE, larger delta -> more like MSE
        """
        super(MaskedHuberLoss, self).__init__()
        self.loss = nn.HuberLoss(reduction='sum', delta=delta)
        self.delta = delta

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch*seq_len]
        
        Huber Loss:
        L = 0.5 * (y - y_hat)^2           if |y - y_hat| <= delta
        L = delta * |y - y_hat| - 0.5 * delta^2   otherwise
        """
        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]
        umask = umask.view(-1, 1) # [batch*seq_len, 1]
        loss = self.loss(pred*umask, target*umask) / torch.sum(umask)
        return loss
# ==================== NEW: Disentangle Loss for Prompt-based Fusion ====================

class JSDLoss(nn.Module):
    """Jensen-Shannon Divergence Loss for aligning shared features"""
    def __init__(self):
        super(JSDLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p, q, umask):
        """
        p, q: [seq_len, batch, dim] - two distributions to compare
        umask: [batch, seq_len]
        """
        p = p.permute(1, 0, 2).contiguous()  # [batch, seq_len, dim]
        q = q.permute(1, 0, 2).contiguous()
        
        p = p.view(-1, p.size(-1))  # [batch*seq_len, dim]
        q = q.view(-1, q.size(-1))
        mask = umask.view(-1, 1)  # [batch*seq_len, 1]
        
        # Apply softmax to convert to probability distributions
        p_prob = F.softmax(p, dim=-1)
        q_prob = F.softmax(q, dim=-1)
        
        m = (0.5 * (p_prob + q_prob)).log()
        jsd = 0.5 * (self.kl(m, p_prob.log()) + self.kl(m, q_prob.log()))
        
        # Apply mask and average
        jsd = (jsd * mask).sum() / max(mask.sum(), 1e-6)
        return jsd


class DisentangleLoss(nn.Module):
    """
    Disentangle Loss to encourage separation between shared and private features
    
    Components:
    1. Private features should be orthogonal to each other (different modalities capture different info)
    2. Private features should be orthogonal to shared features (separation)
    3. Shared features should be aligned (consistency across modalities)
    """
    def __init__(self):
        super(DisentangleLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.jsd = JSDLoss()

    def _masked_abs_cos_sim(self, x, y, umask):
        """
        Calculate masked absolute cosine similarity
        x, y: [seq_len, batch, dim]
        umask: [batch, seq_len]
        """
        x = x.permute(1, 0, 2).contiguous()  # [batch, seq_len, dim]
        y = y.permute(1, 0, 2).contiguous()
        
        x = x.view(-1, x.size(-1))  # [batch*seq_len, dim]
        y = y.view(-1, y.size(-1))
        mask = umask.view(-1)  # [batch*seq_len]
        
        sim = self.cos_sim(x, y).abs()  # [batch*seq_len]
        return (sim * mask).sum() / max(mask.sum(), 1e-6)

    def forward(self, shared_features, private_features, umask):
        """
        Args:
            shared_features: tuple of (shared_a, shared_t, shared_v), each [seq_len, batch, D_e]
            private_features: tuple of (private_a, private_t, private_v), each [seq_len, batch, D_e]
            umask: [batch, seq_len]
        Returns:
            total_loss: scalar
        """
        shared_a, shared_t, shared_v = shared_features
        private_a, private_t, private_v = private_features
        
        # 1. Private features should be orthogonal to each other
        loss_private_ortho = (
            self._masked_abs_cos_sim(private_a, private_t, umask) +
            self._masked_abs_cos_sim(private_a, private_v, umask) +
            self._masked_abs_cos_sim(private_t, private_v, umask)
        ) / 3.0
        
        # 2. Private features should be orthogonal to their corresponding shared features
        loss_private_shared = (
            self._masked_abs_cos_sim(private_a, shared_a, umask) +
            self._masked_abs_cos_sim(private_t, shared_t, umask) +
            self._masked_abs_cos_sim(private_v, shared_v, umask)
        ) / 3.0
        
        # 3. Shared features should be aligned (use JSD to encourage similarity)
        loss_shared_align = (
            self.jsd(shared_a, shared_t, umask) +
            self.jsd(shared_a, shared_v, umask) +
            self.jsd(shared_t, shared_v, umask)
        ) / 3.0
        
        # Total disentangle loss
        total_loss = loss_private_ortho + loss_private_shared + loss_shared_align
        
        return total_loss


class ModalConsistencyLoss(nn.Module):
    """
    Modal Consistency Loss for auxiliary modal-specific predictions
    Encourages modal-specific classifiers to produce consistent predictions
    """
    def __init__(self):
        super(ModalConsistencyLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, log_prob_main, log_prob_a, log_prob_t, log_prob_v, umask):
        """
        All log_probs: [seq_len, batch, n_classes]
        umask: [batch, seq_len]
        """
        # Reshape for KL divergence
        main = log_prob_main.permute(1, 0, 2).contiguous().view(-1, log_prob_main.size(-1))
        a = log_prob_a.permute(1, 0, 2).contiguous().view(-1, log_prob_a.size(-1))
        t = log_prob_t.permute(1, 0, 2).contiguous().view(-1, log_prob_t.size(-1))
        v = log_prob_v.permute(1, 0, 2).contiguous().view(-1, log_prob_v.size(-1))
        
        mask = umask.view(-1, 1)  # [batch*seq_len, 1]
        
        # Convert to probabilities
        main_prob = F.softmax(main, dim=-1)
        a_prob = F.log_softmax(a, dim=-1)
        t_prob = F.log_softmax(t, dim=-1)
        v_prob = F.log_softmax(v, dim=-1)
        
        # KL divergence from main to each modality
        loss = (
            self.kl(a_prob * mask, main_prob * mask) +
            self.kl(t_prob * mask, main_prob * mask) +
            self.kl(v_prob * mask, main_prob * mask)
        ) / 3.0
        
        return loss