from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from extpt.constants import *
    from extpt.datasets import enterface
except ModuleNotFoundError:
    from constants import *
    from datasets import enterface



def cross_entropy_loss_with_centering(teacher_output, student_output, centre):
    teacher_output = teacher_output.detach() # stop gradient
    s = nn.Softmax()(student_output / TEMP_STUDENT)
    t = nn.Softmax()((teacher_output - centre) / TEMP_TEACHER)
    loss = -(t * torch.log(s)).sum().mean()
    if torch.isnan(loss).any():
        ln = torch.log(s)
        mln = t * ln 
        smln = mln.sum()
        msmln = smln.mean()
        nmsmln = -msmln
        breakpoint()
        exit()
    return loss


def multicrop_loss(teacher_outputs: List[torch.tensor] , student_outputs: List[torch.tensor], centre: torch.tensor) -> torch.tensor:
    losses = []
    for teacher_view in teacher_outputs:
        for student_view in student_outputs:
            if not torch.equal(teacher_view, student_view):
                loss = cross_entropy_loss_with_centering(teacher_view, student_view, centre)
                losses.append(loss.unsqueeze(0))
    total_losses = torch.cat(losses)
    return total_losses.mean()


def _contrastive_loss(clip: torch.tensor, clip_prime: torch.tensor) -> torch.tensor:
    batch_sz = clip.shape[0]
    clips_concat = torch.cat((clip_prime, clip))
    numerators = []
    denominators = []
    for i in range(batch_sz):
        z_i = clips_concat[i].repeat((2 * batch_sz, 1))
        sims = torch.exp(F.cosine_similarity(z_i, clips_concat) / TEMP_CONTRASTIVE)
        numerators.append(sims[i].unsqueeze(0))

        # zero out self similarity pair
        mask = torch.ones_like(sims)
        mask[i + batch_sz] = 0
        sims = sims * mask

        denom = sims.sum(dim=0, keepdim=True)
        denominators.append(denom)
    
    numerator_contrastive = torch.cat(numerators)
    denom_contrastive = torch.cat(denominators)
    losses = -torch.log(numerator_contrastive / denom_contrastive)
    losses = losses.mean()
    return losses

def contrastive_loss(clip: torch.tensor, clip_prime: torch.tensor, temp: float) -> torch.tensor:
    batch_sz, emb_sz = clip.shape
    pos_dot = torch.sum(clip * clip_prime, dim=1)
    neg_dots = []
    for i in range(batch_sz):
        nd = torch.zeros(batch_sz).to(DEVICE)
        for j in range(batch_sz):
            if i != j:
                # dot product by hand
                nd[j] = torch.sum(clip[i] * clip_prime[j])
            else:
                nd[j] = pos_dot[j]
        neg_dots.append(nd.unsqueeze(0))
    
    pos_neg_concat = torch.cat(neg_dots, dim=0) 
    logits = pos_neg_concat / temp
    exp = torch.exp(logits)
    exp_diag = torch.diag(exp)
    loss = -torch.log(exp_diag / exp.sum(dim=1))
    return loss.mean()


def get_debiasing_term(x: torch.tensor, x_pos: torch.tensor, temp: float) -> torch.tensor:
    t_plus = 1 / enterface.NUM_LABELS
    t_minus = 1 - t_plus
    x_neg = torch.roll(x_pos, ROLL_N, dims=0)

    negative_sims = torch.sum(x * x_neg, dim=1)
    positive_sims = torch.sum(x * x_pos, dim=1)

    debiasing_term = (negative_sims.mean(dim=1) - (t_plus * positive_sims.mean(dim=1))) / t_minus
    debiasing_term = torch.where(debiasing_term > np.exp(-1/temp), debiasing_term, np.exp(-1/temp))
    return debiasing_term


def contrastive_loss_debiased(z: torch.tensor, z_primes: torch.tensor, temp: float) -> torch.tensor:
    pos_dot = torch.sum(z * z_primes, dim=1)
    logits = pos_dot / temp
    exp = torch.exp(logits)
    loss = -torch.log(exp / exp + get_debiasing_term(z, z_primes, temp))
    print(exp)
    return loss.mean()
        

def get_positives(z: torch.tensor, z_prime: torch.tensor, labels: torch.tensor) -> List[torch.tensor]:
    """
    Returns a list of tensors of shape (NUM_POSITIVES, Z_EMBED_DIM). The list has size BATCH_SZ
    """
    batch_sz, embed_sz = z.shape
    positives_counts = [
        2*labels.bincount().tolist()[i] - 1
        for i in labels.tolist()
    ]

def contrastive_loss_partially_supervised(z: torch.tensor, z_prime: torch.tensor, labels: torch.tensor) -> torch.tensor:
    batch_sz, emb_sz = z.shape
    positives = get_positives(z, z_prime, labels)
    pos_dot = torch.sum(z * z_prime, dim=1)
    neg_dots = []
    for i in range(batch_sz):
        nd = torch.zeros(batch_sz).to(DEVICE)
        for j in range(batch_sz):
            if i != j:
                # dot product by hand
                nd[j] = torch.sum(z[i] * z_prime[j])
            else:
                nd[j] = pos_dot[j]
        neg_dots.append(nd.unsqueeze(0))
    
    pos_neg_concat = torch.cat(neg_dots, dim=0) 
    logits = pos_neg_concat / temp
    exp = torch.exp(logits)
    exp_diag = torch.diag(exp)
    loss = -torch.log(exp_diag / exp.sum(dim=1))
    return loss.mean()
    




# code taken from #https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, ds_namespace, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.constants = ds_namespace
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def _get_cosine_sim(self, batch_1, batch_2, self_contrast_mask):
        assert batch_1.shape == batch_2.shape, "Batches must be the same shape!"
        numerator =  batch_1 @ batch_2.T
        b1_norm = torch.norm(batch_1, dim=1, keepdim=True)
        b2_norm = torch.norm(batch_2, dim=1, keepdim=True)
        denominator = b1_norm @ b2_norm.T
        denominator = torch.where(denominator < self.eps, self.eps, denominator)
        cos = (numerator / denominator) 
        if self_contrast_mask is not None:
            cos = cos * self_contrast_mask
        return cos

    #TODO: write unittests for this
    def _get_labels_for_masking(self, labels, visible_labels_pct):
        label_mask = torch.rand_like(labels.float())
        label_mask = torch.where(label_mask < visible_labels_pct, 1, 0) 
        labels_masked = (labels * label_mask).cuda()
        idx = (labels_masked == 0).nonzero().flatten().view(-1, 1).cuda()
        src = torch.arange(start=self.constants.NUM_LABELS + 1, end=self.constants.NUM_LABELS + 1 + int(idx.shape[0])).view(-1, 1).cuda()
        labels_for_masking = torch.scatter(labels_masked, 0, idx, src)
        labels_for_masking = labels_for_masking.contiguous().view(-1, 1)
        return labels_for_masking
        
    def forward(self, features, labels=None, mask=None, visible_labels_pct=1):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = self._get_labels_for_masking(labels, visible_labels_pct)
            mask = torch.eq(labels, labels.T).float()
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class NeutralAwareLoss(nn.Module):
    def __init__(self, ds_constants, temp=0.05) -> None:
        super().__init__()
        self.temp = temp
        self.eps = 1e-8
        self.scon = SupConLoss(ds_constants)
        self.alpha = 0
        self.beta = 1 - self.alpha

    def _get_cosine_sim(self, batch_1, batch_2, self_contrast_mask):
        assert batch_1.shape == batch_2.shape, "Batches must be the same shape!"
        numerator =  batch_1 @ batch_2.T
        b1_norm = torch.norm(batch_1, dim=1, keepdim=True)
        b2_norm = torch.norm(batch_2, dim=1, keepdim=True)
        denominator = b1_norm @ b2_norm.T
        denominator = torch.where(denominator < self.eps, self.eps, denominator)
        cos = (numerator / denominator) 
        if self_contrast_mask is not None:
            cos = cos * self_contrast_mask
        return cos

        


    def forward(self, neutral_features, emotional_features):
        batch_sz = neutral_features.shape[0]
        self_contrast_mask = 1 - torch.tril(torch.ones(batch_sz, batch_sz)).cuda()
        neutral_similarity = self._get_cosine_sim(neutral_features, neutral_features, self_contrast_mask)
        mean_neutral_logits = neutral_similarity[neutral_similarity != 0].mean()

        cosine_loss = -torch.log(mean_neutral_logits)

        comb_features = torch.cat((emotional_features.unsqueeze(1), emotional_features.unsqueeze(1)), dim=1)
        combined_loss =  (self.alpha * cosine_loss) + (self.beta * self.scon(comb_features, visible_labels_pct=0))
        return combined_loss

        
class HybridLoss(nn.Module):
    def __init__(self, ds_constants):
        super().__init__()
        self.supconloss = SupConLoss(ds_constants)
        self.crossentropyloss = nn.CrossEntropyLoss()
    
    def forward(self, categorical_outputs, features, labels, visible_labels_pct=1):
        s_loss = self.supconloss(features, labels=labels, visible_labels_pct=visible_labels_pct)
        loss = s_loss + self.crossentropyloss(categorical_outputs, labels.flatten())
        return loss
    
class HybridLossNeutralAware(nn.Module):
    def __init__(self, ds_constants):
        super().__init__()
        self.naloss = NeutralAwareLoss(ds_constants)
        self.crossentropyloss = nn.CrossEntropyLoss()
    
    def forward(self, categorical_outputs, neutral_features, emotional_features, labels):
        n_loss = self.naloss(neutral_features, emotional_features)
        loss = n_loss + self.crossentropyloss(categorical_outputs, labels.flatten())
        return loss
        
class AVDGuidedContrastiveLoss(nn.Module):
    def __init__(self, ds_constants):
        super().__init__()
        self.supconloss = SupConLoss(ds_constants)
        self.mse = nn.MSELoss()
        self.alpha = 0.5
    
    def forward(self, avd_preds, emotional_features, avd_labels):
        s_loss = self.supconloss(emotional_features)
        loss = self.alpha * s_loss + ((1 - self.alpha) * self.mse(avd_preds, avd_labels))
        return loss

if __name__ == "__main__":
    from timeit import timeit
    from torch.nn.functional import normalize
    batch_sz = 32
    temp = 0.07
    M = 3
    l = NeutralAwareLoss()


    # p = torch.rand((batch_sz, 3))
    # q = torch.rand((batch_sz, 3))

    a = torch.tensor([
        [-0.83483301, -0.16904167, 0.52390721],
        [ 0.70374682, -0.18682394, -0.68544673],
        [ 0.15465702,  0.32303224,  0.93366556],
        [ 0.53043332, -0.83523217, -0.14500935],
        [ 0.68285685, -0.73054075,  0.00409143],
        [ 0.76652431,  0.61500886,  0.18494479]
    ])
    b = torch.tensor([
        [-0.83455951, -0.16862266, 0.52447767],
        [ 0.70374682, -0.18682394, -0.68544673],
        [ 0.15465702,  0.32303224,  0.93366556],
        [ 0.53043332, -0.83523217, -0.14500935],
        [ 0.68285685, -0.73054075,  0.00409143],
        [ 0.76652431,  0.61500886,  0.18494479]
    ])
    loss = l(a, b)
    print(loss)

    exit()
    z = normalize(torch.rand((batch_sz, 3)), dim=1)
    z_primes = z.clone().repeat(M, 1, 1)


    print(contrastive_loss_debiased(z, z_primes, temp))
    # print(contrastive_loss_debiased(p, q, temp))
    # print(contrastive_loss(a, b, temp))
    # print(contrastive_loss(p, q, temp))
    # print(contrastive_loss(a, b, temp))
    # print(contrastive_loss(p, q, temp))
    exit()
    timeit_globals = {
        "contrastive_loss": contrastive_loss,
        "naive_contrastive_loss": naive_contrastive_loss,
        "z": a,
        "z_prime": b,
        "temp": 0.07
    }
    N = 100_000
    print(timeit('contrastive_loss(z, z_prime, temp)', number=N, globals=timeit_globals))
    exit()
    g1 = torch.rand(5)
    g2 = torch.rand(5)
    l1 = torch.rand(5)
    l2 = torch.rand(5)
    l3 = torch.rand(5)
    centre = torch.rand(5)


    t_vs = [g1, g2]
    s_vs = [g1, g2, l1, l2, l3]
    
    total_loss = multicrop_loss(t_vs, s_vs, centre)
    print(total_loss)
