import math
import datetime
import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
import numpy as np
import sys
from model.DataSet import ReMITDataSet
from torch.nn import Linear
import torch.nn.functional as F

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class ReMIT:
    def __init__(self, args):
        self.args = args
        self.the_data = args.dataset
        self.file_path = '../data/%s/%s.txt' % (self.the_data, self.the_data)
        self.emb_path = '../emb/%s/%s_ReMIT_%d.emb'
        self.label_path = '../data/%s/label.txt' % (self.the_data)
        self.feature_path = './pretrain/%s_SDCN.emb' % self.the_data
        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.cluster
        self.save_step = args.save_step
        self.epochs = args.epoch
        self.lambdas = args.lambdas

        self.data = ReMITDataSet(args, self.file_path, self.neg_size, self.hist_len, self.feature_path, args.directed)
        self.node_dim = self.data.get_node_dim()
        self.edge_num = self.data.get_edge_num()
        self.feature = self.data.get_feature()

        if torch.cuda.is_available(): 
            with torch.cuda.device(DID):
                '''
                position_emb = ReMIT.position_encoding_(self, self.node_list, self.emb_size)
                self.node_emb = Variable(position_emb.cuda(), requires_grad=True)'''
                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, self.emb_size))).type(
                    FType).cuda(), requires_grad=True)
                # self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=True)
                self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=False)
                self.global_emb = Variable(torch.mean(self.pre_emb, dim=0).type(FType).cuda(), requires_grad=True)
                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.lambda_g = Variable((torch.zeros(1, self.emb_size) + 0.5).type(FType).cuda(), requires_grad=True)
                self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cuda(),
                                              requires_grad=True)
                torch.nn.init.xavier_normal_(self.cluster_layer.data)

                kmeans = KMeans(n_clusters=self.clusters, n_init=20)
                _ = kmeans.fit_predict(self.feature)
                self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

                self.v = 1.0

        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb, self.global_emb, self.delta, self.lambda_g, self.cluster_layer])
        self.loss = torch.FloatTensor()

    def kl_loss(self, z, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')  # l_clu
        return the_kl_loss

    def target_dis(self, emb):
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        tmp_q = q.data
        weight = tmp_q ** 2 / tmp_q.sum(0)
        p = (weight.t() / weight.sum(1)).t()

        return p

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):

        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_pre_emb = self.pre_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        global_emb = self.global_emb

        s_p = self.target_dis(s_pre_emb)
        l_com = self.kl_loss(s_node_emb, s_p)

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(
            dim=1)  # [b]

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()
        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        l_tem = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(
            n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)  # [b]

        align_diff_x = ((s_node_emb - s_pre_emb) ** 2).sum(dim=1).neg()
        align_diff_y = ((t_node_emb - t_pre_emb) ** 2).sum(dim=1).neg()
        l_n = -torch.log(align_diff_x.sigmoid() + 1e-6) - torch.log(align_diff_y.sigmoid() + 1e-6)

        l_global = -torch.log(((s_node_emb - global_emb) ** 2).sum(dim=1).neg().sigmoid() + 1e-6) - torch.log(
            ((t_node_emb - global_emb) ** 2).sum(dim=1).neg().sigmoid() + 1e-6)
        self.global_emb.data = global_emb + torch.mean(s_node_emb, dim=0) + torch.mean(t_node_emb, dim=0)
        total_loss = l_tem.sum() + self.lambdas * (l_n.sum() + l_com + l_global.sum())
        # total_loss = l_tem.sum() + self.lambdas * (l_n.sum() + l_com + l_global.sum())

        return total_loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            # start = datetime.datetime.now()
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=1)

            # if epoch % self.save_step == 0 and epoch != 0:
            if epoch == 10:
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, epoch))
            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, epoch))
            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))

            sys.stdout.write('\repoch %d: loss=%.4f\n' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
            # end = datetime.datetime.now()
            # print('Training Complete with Time: %s' % str(end - start))

            sys.stdout.flush()

        self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, self.epochs))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()
