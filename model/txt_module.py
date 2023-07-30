import torch
import torch.nn.init as init
from .basic_module import BasicModule
import torch.nn as nn
import torch.nn.functional as F
from GMCH.util import gen_A
from GMCH.util import gen_adj
from torch.nn import Parameter
from GMCH.model.gcn_net import GraphConvolution, GraphAttentionLayer

class TxtModule(nn.Module):
    """Network to learn text representations"""

    def __init__(self, dropout, hidden_dim, bit, input_dim=1386, output_dim=4096, batch_size=128,
                 num_classes=24, in_channel=300, t=0, adj_file=None, inp=None, GNN='GAT', n_layers=4):
        super(TxtModule, self).__init__()
        self.bit = bit
        # self.cnn_f = image_net(pretrain_model)   ## if use 4096-dims feature, pass
        if dropout:
            self.text_module = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
                nn.Dropout(0.5),
            )
        else:
            self.text_module = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True)
            )

        self.hash_module = nn.ModuleDict({
            'image': nn.Sequential(
                nn.Linear(hidden_dim // 4, bit, bias=True),
                nn.Tanh()),
            'text': nn.Sequential(
                nn.Linear(hidden_dim // 4, bit, bias=True),
                nn.Tanh()),
        })

        if GNN == 'GAT':
            self.gnn = GraphAttentionLayer
        elif GNN == 'GCN':
            self.gnn = GraphConvolution
        else:
            raise NameError("Invalid GNN name!")
        self.n_layers = n_layers

        self.relu = nn.LeakyReLU(0.2)
        self.lrn = [self.gnn(in_channel, output_dim)]
        for i in range(1, self.n_layers):
            self.lrn.append(self.gnn(output_dim, output_dim))
        for i, layer in enumerate(self.lrn):
            self.add_module('lrn_{}'.format(i), layer)
        self.hypo = nn.Linear(self.n_layers * output_dim, output_dim)

        _adj = torch.FloatTensor(gen_A(num_classes, t, adj_file))
        if GNN == 'GAT':
            self.adj = Parameter(_adj, requires_grad=False)
        else:
            self.adj = Parameter(gen_adj(_adj), requires_grad=False)

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def weight_init(self):
        initializer = self.kaiming_init
        for block in self._modules:
            if block == 'cnn_f':
                pass
            else:
                for m in self._modules[block]:
                    initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, y):
        # x = self.cnn_f(x).squeeze()   ## if use 4096-dims feature, pass

        f_y = self.text_module(y)
        y_code = self.hash_module['text'](f_y).reshape(-1, self.bit)

        layers = []
        x = self.inp
        for i in range(self.n_layers):
            x = self.lrn[i](x, self.adj)
            if self.gnn == GraphConvolution:
                x = self.relu(x)
            layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)
        norm_txt = torch.norm(f_y, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        x = x.transpose(0, 1)
        y_text = torch.matmul(f_y, x)
        y_text = y_text / norm_txt

        return y_text, y_code


    def generate_txt_code(self, t):
        f_t = self.text_module(t)
        code = self.hash_module['text'](f_t.detach()).reshape(-1, self.bit)

        return code



