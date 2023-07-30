from __future__ import print_function
from __future__ import division
import time
import copy
from torch.utils.data import DataLoader
from data_loader import Dataset, loader
from tqdm import tqdm
import numpy as np
import torch
from evaluate import calc_map_k2, generate_img_code2, generate_txt_code2
from loss import cla_loss,  calc_loss, mdl_loss, calc_label_sim
# 训练模块
def train_model(model_img, model_txt, optimizer_img, optimizer_txt, max_epoch, batch_size,
            bit, gamma, eta, alpha, dataset, db_size, training_size, query_size, DATA_DIR):
    since = time.time()


    # 提取必要的输入参数
    '''img_test = input_data_par['img_test']
    text_test = input_data_par['text_test']
    label_test = input_data_par['label_test']
    img_train = input_data_par['img_train']
    text_train = input_data_par['text_train']
    label_train = input_data_par['label_train']
    img_dim = input_data_par['img_dim']
    text_dim = input_data_par['text_dim']
    num_class = input_data_par['num_class']
    img_num = input_data_par['img_num']'''

    images, tags, labels = loader(DATA_DIR, dataset)
    train_data = Dataset(dataset, db_size, training_size, query_size, images, tags, labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    L = train_data.get_labels()

    i_query_data = Dataset(dataset, db_size, training_size, query_size, images, tags, labels, test='image.query')
    i_db_data = Dataset(dataset, db_size, training_size, query_size, images, tags, labels, test='image.db')
    t_query_data = Dataset(dataset, db_size, training_size, query_size, images, tags, labels, test='text.query')
    t_db_data = Dataset(dataset, db_size, training_size, query_size, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, batch_size, shuffle=False)
    i_db_dataloader = DataLoader(i_db_data, batch_size, shuffle=False)
    t_query_dataloader = DataLoader(t_query_data, batch_size, shuffle=False)
    t_db_dataloader = DataLoader(t_db_data, batch_size, shuffle=False)

    query_labels, db_labels = i_query_data.get_labels()
    query_labels = query_labels.cuda()
    db_labels = db_labels.cuda()

    # 进行参数操作
    #label_train = torch.from_numpy(L)
    label_train = L
    F_buffer = torch.randn(training_size, bit)
    G_buffer = torch.randn(training_size, bit)
    if torch.cuda.is_available():
        label_train = label_train.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()

    Sim = calc_label_sim(label_train, label_train)
    B = torch.sign(F_buffer + G_buffer)
    ones = torch.ones(batch_size, 1)
    ones_ = torch.ones(training_size - batch_size, 1)



    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_img.to(device)
    model_txt.to(device)
    best_acc = 0.0
    best_model_img_wts = copy.deepcopy(model_img.state_dict())
    best_model_txt_wts = copy.deepcopy(model_txt.state_dict())
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []

    # 训练循环

    for epoch in range(max_epoch):
        print('Epoch {}/{}'.format(epoch + 1, max_epoch))
        print('-' * 20)
        running_loss = 0.0
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                model_img.train()
                model_txt.train()
            else:
                # Set model to evaluate mode
                model_img.eval()
                model_txt.eval()
        # 第一个模块的训练循环

            for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
            #for i in tqdm(range(training_size // batch_size), desc='Epoch {}/{}'.format(1, img_num // batch_size), unit='batch'):

                #index = np.random.permutation(training_size)
                #ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(training_size), ind)
                #sample_L = label_train[ind, :]
                #print(ind.min(), ind.max())  # 打印索引的最小值和最大值
                #print(img.shape[0])  # 打印数组 img 的维度大小
                #image = torch.from_numpy(img[ind]).type(torch.float)
                if i == len(train_dataloader) - 1:
                    batch_size1 = len(ind)  # 最后一个批次的样本数量
                    ones1 = torch.ones(batch_size1, 1).cuda()
                    ones1_ = torch.ones(training_size - batch_size1, 1).cuda()
                    denom = batch_size1 * training_size
                else:
                    ones1 = ones.cuda()
                    ones1_ = ones_.cuda() # 其他批次的样本数量
                    denom = batch_size * training_size



                if torch.cuda.is_available():
                    image = img.cuda()
                    sample_L = label.cuda()


                    # similar matrix size: (batch_size, num_train)
                S = calc_label_sim(sample_L, label_train)  # S: (batch_size, num_train)
                img_score, cur_f = model_img(image)  # cur_f: (batch_size, bit)
                F_buffer[ind, :] = cur_f.data
                F = F_buffer
                G = G_buffer

                theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
                logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
                quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))
                balance_x = torch.sum(torch.pow(cur_f.t().mm(ones1) + F[unupdated_ind].t().mm(ones1_), 2))
                clas_loss_x = cla_loss(img_score, sample_L)
                loss_x = logloss_x + gamma * quantization_x + eta * balance_x
                loss_x = loss_x.mean() + alpha * clas_loss_x


                optimizer_img.zero_grad()
                loss_x.backward()
                optimizer_img.step()



            # 第二个模块的训练循环
            for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
            #for i in tqdm(range(img_num // batch_size), desc='Epoch {}/{}'.format(1, img_num // batch_size), unit='batch'):
                    # 模块2的训练逻辑
                #index = np.random.permutation(training_size)
                #ind = index[0: batch_size]
                unupdated_ind = np.setdiff1d(range(training_size), ind)
                #sample_L = label_train[ind, :]
                #text = torch.from_numpy(txt[ind, :]).type(torch.float)
                if i == len(train_dataloader) - 1:
                    batch_size2 = len(ind)  # 最后一个批次的样本数量
                    ones2 = torch.ones(batch_size2, 1).cuda()
                    ones2_ = torch.ones(training_size - batch_size2, 1).cuda()
                    denom2 = batch_size2 * training_size
                else:
                    ones2 = ones.cuda()
                    ones2_ = ones_.cuda()  # 其他批次的样本数量
                    denom2 = batch_size * training_size

                if torch.cuda.is_available():
                    text = txt.cuda()
                    sample_L = label.cuda()

                    # similar matrix size: (batch_size, num_train)
                S = calc_label_sim(sample_L, label_train)  # S: (batch_size, num_train)
                txt_score, cur_g = model_txt(text)  # cur_f: (batch_size, bit)
                G_buffer[ind, :] = cur_g.data
                F = F_buffer
                G = G_buffer

                    # calculate loss
                    # theta_y: (batch_size, num_train)
                theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
                logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
                quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))


                balance_y = torch.sum(torch.pow(cur_g.t().mm(ones2) + G[unupdated_ind].t().mm(ones2_), 2))
                clas_loss_y = cla_loss(txt_score, sample_L)
                loss_y = logloss_y + gamma * quantization_y + eta * balance_y
                loss_y = loss_y.mean() + alpha * clas_loss_y



                optimizer_txt.zero_grad()
                loss_y.backward()
                optimizer_txt.step()

            B = torch.sign(F_buffer + G_buffer)

            # calculate total loss
            loss = calc_loss(B, F, G, Sim, gamma, eta) + alpha * (clas_loss_x + clas_loss_y)



            if phase == 'train':
                print('Train Loss: {:.7f}'.format(loss.item()))
            if phase == 'test':
                with torch.no_grad():


                    qBX = generate_img_code2(model_img, i_query_dataloader, query_size, bit, batch_size)
                    qBY = generate_txt_code2(model_txt, t_query_dataloader, query_size, bit, batch_size)
                    rBX = generate_img_code2(model_img, i_db_dataloader, db_size, bit, batch_size)
                    rBY = generate_txt_code2(model_txt, t_db_dataloader, db_size, bit, batch_size)

                    img2txt = calc_map_k2(qBX, rBY, query_labels, db_labels)
                    txt2img = calc_map_k2(qBY, rBX, query_labels, db_labels)

                    print('{} Loss: {:.7f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, max_epoch, img2txt, txt2img))

            if phase == 'test' and (img2txt + txt2img) / 2. > best_acc:
                best_acc = (img2txt + txt2img) / 2.
                best_model_img_wts = copy.deepcopy(model_img.state_dict())
                best_model_txt_wts = copy.deepcopy(model_txt.state_dict())
            if phase == 'test':

                test_img_acc_history.append(img2txt)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(loss)

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))
# 保存最好的模型参数
    model_img.load_state_dict(best_model_img_wts)
    model_txt.load_state_dict(best_model_txt_wts)

# 保存模型参数

    return model_img, model_txt, test_img_acc_history, test_txt_acc_history, epoch_loss_history



