import os
import torch
import torch.optim as optim
from evaluate import calc_map_k2, generate_img_code2, generate_txt_code2
from scipy.io import loadmat
from model.img_module import ImgModule
from model.txt_module import TxtModule
from train_model import train_model
from torch.utils.data import DataLoader
from data_loader import Dataset, loader



######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset = 'mirflickr'  # 'mirflickr' or 'NUS-WIDE-TC21' or 'MS-COCO'
    model = 'P-GNN'  # 'I-GNN' or 'P-GNN'
    embedding = 'glove'  # 'glove' or 'googlenews' or 'fasttext' or 'None'
    bit = 64

    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    EVAL = False    # True for evaluation, False for training
    INCOMPLETE = False   # True for incomplete-modal learning, vice versa
    dropout = False
    if dataset == 'mirflickr':
        alpha = 0.4
        beta = 1
        gamma = 1
        eta = 0.1
        max_epoch = 100
        batch_size = 128
        hidden_dim = 16384
        lr = 10 ** (-5)
        lr2 = 10 ** (-1.5)
        betas = (0.5, 0.9)
        t = 0.4
        gnn = 'GCN'  # 'GCN' or 'GAT'
        n_layers = 2   # number of GNN layers
        k = 8
        db_size = 18015
        num_label = 24
        query_size = 2000
        text_dim = 1386
        training_size = 10000

    elif dataset == 'NUS-WIDE-TC21':
        alpha = 0.4
        beta = 0.2
        gamma = 1
        eta = 0.1
        max_epoch = 50
        batch_size = 2048
        hidden_dim = 16384
        lr = 5e-5
        lr2 = 1e-8
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 2
        k = 8
        db_size = 193734
        num_label = 21
        query_size = 2100
        text_dim = 1000
        training_size = 10000

    elif dataset == 'MS-COCO':
        alpha = 2.8
        beta = 0.2
        gamma = 1
        eta = 1
        max_epoch = 500
        batch_size = 128
        hidden_dim = 8192
        lr = 5e-5
        lr2 = 1e-7
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 2
        k = 8

    else:
        raise NameError("Invalid dataset name!")


    if embedding == 'glove':
        inp = loadmat('embedding/' + dataset + '-inp-glove6B.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'googlenews':
        inp = loadmat('embedding/' + dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'fasttext':
        inp = loadmat('embedding/' + dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
    else:
        inp = None

    print('...Data loading is beginning...')



    print('...Data loading is completed...')

    if model == 'P-GNN':
        model_img = ImgModule(dropout=dropout, hidden_dim=hidden_dim, bit=bit, img_input_dim=4096, output_dim=4096,
                              batch_size=batch_size, num_classes=num_label, t=t, adj_file='data/' + dataset + '/adj.mat', inp=inp,
                         GNN=gnn, n_layers=n_layers).cuda()
        model_txt = TxtModule(dropout=dropout, hidden_dim=hidden_dim, bit=bit, input_dim=text_dim, output_dim=4096,
                              batch_size=batch_size, num_classes=num_label, t=t, adj_file='data/' + dataset + '/adj.mat', inp=inp,
                         GNN=gnn, n_layers=n_layers).cuda()
    else:
        raise NotImplementedError("The model should be 'I-GNN' or 'P-GNN'.")

    params_to_update_img = list(model_img.parameters())
    params_to_update_txt = list(model_txt.parameters())

    # Observe that all parameters are being optimized
    optimizer_img = torch.optim.Adam(params_to_update_img, lr=lr, betas=betas, weight_decay=0.0005)
    optimizer_txt = torch.optim.Adam(params_to_update_txt, lr=lr, betas=betas, weight_decay=0.0005)#这里是用的公式优化算法
    if EVAL:
        model_img.load_state_dict(torch.load('/root/autodl-tmp/project/GMCH/model/img_' + dataset + str(bit) + '.pth'))
        model_txt.load_state_dict(torch.load('/root/autodl-tmp/project/GMCH/model/txt_' + dataset + str(bit) + '.pth'))
    else:
        print('...Training is beginning...')
        # Train and evaluate

        model_img, model_txt, test_img_acc_history, test_txt_acc_history, epoch_loss_history = train_model(
            model_img, model_txt, optimizer_img, optimizer_txt, max_epoch, batch_size,
            bit, gamma, eta, alpha, dataset, db_size, training_size, query_size, DATA_DIR)
        print('...Training is completed...')

        torch.save(model_img.state_dict(), 'model/img_' + dataset + str(bit) + '.pth')
        torch.save(model_txt.state_dict(), 'model/txt_' + dataset + str(bit) + '.pth')

    print('...Evaluation on testing data...')
    model_img.eval()
    model_txt.eval()

    images, tags, labels = loader(DATA_DIR, dataset)
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

    qBX = generate_img_code2(model_img, i_query_dataloader, query_size, bit, batch_size)
    qBY = generate_txt_code2(model_txt, t_query_dataloader, query_size, bit, batch_size)
    rBX = generate_img_code2(model_img, i_db_dataloader, db_size, bit, batch_size)
    rBY = generate_txt_code2(model_txt, t_db_dataloader, db_size, bit, batch_size)


    img2txt = calc_map_k2(qBX, rBY, query_labels, db_labels)
    print('...Image to Text MAP = {}'.format(img2txt))

    txt2img = calc_map_k2(qBY, rBX, query_labels, db_labels)
    print('...Text to Image MAP = {}'.format(txt2img))

    print('...Average MAP = {}'.format(((img2txt + txt2img) / 2.)))



