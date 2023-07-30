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





