import numpy as np
import torch
from torch.autograd._functions import tensor
from tqdm import tqdm



def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH





def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    # qB:查询集  范围{-1,+1}
    # rB:检索集  范围{-1,+1}
    # query_label: 查询标签
    # retrieval_label: 检索标签
    num_query = query_label.shape[0]  #查询个数
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]  #如果不指定k,k将是全部检索个数。对于flickr25k数据集，即18015
    for iter in range(num_query):
        #每个查询标签乘以检索标签的转置，只要有相同标签，该位置就是1
        gnd = (torch.from_numpy(query_label[iter]).unsqueeze(0).mm(torch.from_numpy(retrieval_label).t()) > 0).type(torch.float).squeeze()
        if torch.cuda.is_available():
            gnd=gnd.cuda()
        tsum = torch.sum(gnd)   #真实相关的数据个数
        #print("相关个数：", tsum)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm) #ind ：已排序的汉明距，在未排序中的位置
        ind.squeeze_()
        #print("原始 gnd:", gnd)
        #print("ind    :", ind)
        gnd = gnd[ind]  #按照预测的顺序重排
        #print("重排后gnd:", gnd)
        total = min(k, int(tsum))  #取k和tsum的最小值，这句应该没啥用
        #如果有三个相关的，则count是[1，2，3]
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        #取出重排后非0元素的位置
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        x=count.shape
        y=tindex.shape
        #print("count:", x)
        #print("tindex:", y)
        map += torch.mean(count / tindex)
        #print("map:", map)
    map = map / num_query
    return map





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



def calc_map_k2(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map

def generate_img_code2(model, test_dataloader, num, bit, batch_size):
    B = torch.zeros(num, bit).cuda()

    for i, input_data in tqdm(enumerate(test_dataloader)):
        input_data = input_data.cuda()
        b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * batch_size)
        B[i * batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B

def generate_txt_code2(model, test_dataloader, num, bit, batch_size):
    B = torch.zeros(num, bit).cuda()

    for i, input_data in tqdm(enumerate(test_dataloader)):
        input_data = input_data.cuda()
        b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * batch_size)
        B[i * batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B