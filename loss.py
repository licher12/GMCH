import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm



def calc_label_sim(label_1, label_2):
    if isinstance(label_1, np.ndarray):
        label_1 = torch.from_numpy(label_1).float().cuda()
    elif isinstance(label_1, torch.Tensor):
        label_1 = label_1.float().cuda()

    if isinstance(label_2, np.ndarray):
        label_2 = torch.from_numpy(label_2).float().cuda()
    elif isinstance(label_2, torch.Tensor):
        label_2 = label_2.float().cuda()

    Sim = label_1.mm(label_2.t())
    return Sim




def cla_loss(view1_predict, labels_1):

    loss = ((view1_predict - labels_1.float()) ** 2).sum(1).mean()


    return loss





def mdl_loss(view1_feature, view2_feature, labels_1, labels_2):
    cos = lambda x, y: x.mm(y.t()) / (
        (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term11 = ((1 + torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term12 = ((1 + torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term22 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    mdl_loss = term11 + term12 + term22

    return mdl_loss
def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

'''def soft_con_loss(view1_feature, view2_feature, labels, t=0.21, gamma=0.13):
    view1_feature = F.normalize(view1_feature, dim=1)
    view2_feature = F.normalize(view2_feature, dim=1)
    # cosine similarity: NxN（余弦相似性）
    sim_view12 = torch.matmul(view1_feature, view2_feature.T) / t#两个张量的乘积，这里把两个标签，如果两输入的量为一维则进行点乘，二维就是矩阵相乘
    sim_view11 = torch.matmul(view1_feature, view1_feature.T) / t
    sim_view22 = torch.matmul(view2_feature, view2_feature.T) / t
    #label_L1 = labels.sum(1)
    #label_sim = torch.matmul(labels, labels.T) / (label_L1[None, :] + label_L1[:, None] - torch.matmul(labels, labels.T))
    label_sim = torch.matmul(labels, labels.T).clamp(max=1.0)
    #label_sim = label_sim ** 0.5
    pro_inter = label_sim / label_sim.sum(1, keepdim=True).clamp(min=1e-6)
    label_sim_intra = (label_sim - torch.eye(label_sim.shape[0]).cuda()).clamp(min=0)
    pro_intra = label_sim_intra / label_sim_intra.sum(1, keepdim=True).clamp(min=1e-6)

    # logits: NxN
    logits_view12 = sim_view12 - torch.log(torch.exp(1.06 * sim_view12).sum(1, keepdim=True))
    logits_view21 = sim_view12.T - torch.log(torch.exp(1.06 * sim_view12.T).sum(1, keepdim=True))
    logits_view11 = sim_view11 - torch.log(torch.exp(1.06 * sim_view11).sum(1, keepdim=True))
    logits_view22 = sim_view22 - torch.log(torch.exp(1.06 * sim_view22).sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos_view12 = (pro_inter * logits_view12).sum(1)
    mean_log_prob_pos_view21 = (pro_inter * logits_view21).sum(1)
    mean_log_prob_pos_view11 = (pro_intra * logits_view11).sum(1)
    mean_log_prob_pos_view22 = (pro_intra * logits_view22).sum(1)

    # supervised cross-modal contrastive loss
    loss = - mean_log_prob_pos_view12.mean() - mean_log_prob_pos_view21.mean() \
           - gamma * (mean_log_prob_pos_view11.mean() + mean_log_prob_pos_view22.mean())

    return loss'''



def generate_image_code(img_model, X, bit, batch_size):
    batch_size = batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if torch.cuda.is_available():
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = torch.from_numpy(X[ind]).type(torch.float)
        if torch.cuda.is_available():
            image = image.cuda()
        _, cur_f = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit, batch_size):
    batch_size = batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if torch.cuda.is_available():
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = torch.from_numpy(Y[ind, :]).type(torch.float)
        if torch.cuda.is_available():
            text = text.cuda()
        _, cur_g = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B



def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss



'''class TripletHardLoss(nn.Module):
    def __init__(self, dis_metric='euclidean', squared=False, reduction='mean'):
        """
        Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        :param margin:
        :param dis_metric: 'euclidean' or 'dp'(dot product)
        :param squared:
        :param reduction: 'mean' or 'sum' or 'none'
        """
        super(TripletHardLoss, self).__init__()

        self.dis_metric = dis_metric
        self.reduction = reduction
        self.squared = squared

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels


        pairwise_dist = cos_distance(source, target)

        # First, we need to get a mask for every valid positive (they should have same label)
        # and every valid negative (they should have different labels)
        mask_anchor_positive, mask_anchor_negative = _get_anchor_triplet_mask(s_labels, t_labels)

        # For each anchor, get the hardest positive
        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)

        # For each anchor, get the hardest negative
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(dim=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

        # Get final mean triplet loss
        if self.reduction is 'mean':
            triplet_loss = triplet_loss.mean()
        elif self.reduction is 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss'''



