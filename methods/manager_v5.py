%%writefile methods/manager.py

from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder
from .utils import Moment, dot_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from .utils import osdist
class Manager(object):
    def __init__(self, args):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
    def get_proto(self, args, encoder, mem_set):
        # aggregate the prototype set for further use.
        data_loader = get_data_loader(args, mem_set, False, False, 1)

        features = []

        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature, rep= encoder.bert_forward(tokens)
            features.append(feature)
            self.lbs.append(labels.item())
        features = torch.cat(features, dim=0)

        proto = torch.mean(features, dim=0, keepdim=True)

        return proto, features
    # Use K-Means to select what samples to save, similar to at_least = 0
    def select_data(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens=torch.stack([x.to(args.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())

        features = np.concatenate(features)
        num_clusters = min(args.num_protos, len(sample_set))
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

        mem_set = []
        current_feat = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
            current_feat.append(features[sel_index])
        
        current_feat = np.stack(current_feat, axis=0)
        current_feat = torch.from_numpy(current_feat)
        return mem_set, current_feat, current_feat.mean(0)
    
    def get_optimizer(self, args, encoder):
        print('Use {} optim!'.format(args.optim))
        def set_param(module, lr, decay=0):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ['undecay']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize
        params = set_param(encoder, args.learning_rate)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer
    def train_simple_model(self, args, encoder, training_data, epochs):

        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()

        optimizer = self.get_optimizer(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                hidden, reps = encoder.bert_forward(tokens)
                loss = self.moment.loss(reps, labels)
                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach(), hidden.detach())
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i), is_mem=False)
            
 
    @torch.no_grad()
    def get_concentration(self, args, encoder, training_data, protos_raw, current_relations):
        """
            inputs: 
                + encoder: nn.Module()
                + training_data: dict[relation, [labels, tokens, ind]]
                + protos_raw: list[torch.Tensor()]
            return:
                + concentraition: dict[relation, dist[torch.Tensor(n_seen_relations)]]
        """
        
        protos_raw = torch.cat(protos_raw, dim = 0)
        concentration = {}
        for rel in current_relations:
            train_data_per_rel = []
            train_data_per_rel += training_data[rel]
            data_loader = get_data_loader(args, train_data_per_rel, batch_size = 64)
            
            hiddens = []
            for step, batch_data in enumerate(data_loader):
                labels, tokens, ind = batch_data
                tokens = torch.stack([x.to(args.device) for x in tokens], dim = 0)
                hidden, _ =  encoder.bert_forward(tokens)
                hiddens.append(hidden)
            hiddens = torch.cat(hiddens, dim = 0)
            
            # expect: samples : n_samples x hidden_dim
            dist = torch.cdist(hiddens.unsqueeze(0), protos_raw.unsqueeze(0).to(args.device), p = 2).squeeze(0)
            # expect: n_samples x n_seen_relations
            dist = torch.mean(dist, dim = 0)
            concentration[self.rel2id[rel]] = dist
        return concentration
    
    def get_protoNCE_loss(self, args, hiddens, protos_raw, balance_class, seen_relations, labels, concentraion):
        """
            inputs: 
                + hiddens: torch.Tensor(requires_grad = True) : batch_size x 768
                + protos_raw: torch.Tensor(requires_grad = True) : n_seen_relations x 768 
                + balance_class: torch.Tenor(): [1 x batch_size]
                + seen_relations: list[str]
                + labels: torch.Tensor(): 1 x len(current_relations)
                + concentration : dict[relation; torch.Tensor([1 x len(seen_relations)])]
            outputs:
                + protoNCE_loss: torch.tensor(requires_grad = True)
        """
        phi = []
        hiddens = F.normalize(hiddens, p = 2, dim = 1)
#         protos_raw = [p.reshape(1, 768) for p in protos_raw]            
        protos_raw = torch.cat(protos_raw, dim =0)
        protos_raw = F.normalize(protos_raw, p = 2, dim = 1)
        
        for rel in labels:
            phi.append(concentraion[(int)(rel.item())])
        phi = torch.stack(phi, dim = 0)
#          batch_size x n_seen_relation
        # step 1: calculation contrastive loss between hiddens and protos_raw
        protos_raw = protos_raw.to(args.device)
        dot_product = torch.mm(hiddens, protos_raw.T)
        # expect: dot_product : batch_size x n_seen_relations 
        dot_product *= phi
        dot_product = torch.exp(dot_product - torch.max(dot_product, dim = 1, keepdim = True)[0].detach()) + 1e-5
        labels = labels.to(args.device)
        seen_relations = torch.Tensor([self.rel2id[i] for i in seen_relations]).to(args.device)
        mask = labels.unsqueeze(1).repeat(1, seen_relations.shape[0]) == seen_relations
        prob = -torch.log(dot_product / torch.sum(dot_product, dim = 1, keepdim = True))
        contrastive_loss = torch.sum(prob*mask, dim = 1) * balance_class
        contrastive_loss = torch.mean(contrastive_loss)

        return contrastive_loss
    def update_protos(self, protos_raw, protos_index, protos_hidden_len, hiddens, ind, labels, relation):
        """
        input: 
            + protos_raw: torch.Tensor() : len_seen_relations x args.encoder_bert_out
            + hiddens : batch_size x args.encoder_bert_out
            + relation: str 
            + protos_hidden_len : list[torch.tensor(int)]
        return:
            + new_protos = torch.Tensor(): 1 x args.encoder_bert_out
        """
        relation_idx = np.argmax(np.array(protos_index) == self.rel2id[relation])
        
        old_feat_len = protos_hidden_len[relation_idx]
        old_feat_len = old_feat_len.item()
        old_proto = protos_raw[relation_idx]
        
        mask = torch.Tensor(np.array(labels.to('cpu')) == self.rel2id[relation]).cuda()
        mask = mask.reshape(1, -1)

        current_proto = torch.matmul(mask , hiddens)
        current_feat_len = torch.sum(mask).item()
        if current_feat_len == 0:
            return old_proto, relation_idx
        update_proto = (old_feat_len * old_proto + current_feat_len * current_proto)/(old_feat_len + current_feat_len)
        update_proto = update_proto.reshape(1, -1)
        update_proto = update_proto.to('cuda:0')
        return update_proto, relation_idx
        

    def get_optimizer_L2_norm(self, args, encoder):
        print('Use {} optim!'.format(args.optim))
        def set_param(module, lr, decay=0):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ['undecay']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize
        params = set_param(encoder, 0.1)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer
    def train_no_name_model(self, args, encoder, training_data, protos_raw, seen_relations, current_relations, proto_dict,protos_index,protos_hidden_len, concentration):
        """
            input: 
                + training_data: list
                + protos_raw: list[torch.Tensor([1 x encoderoutput_dim])]
                + seen_relations : list[str]
                + current_relations : list[str]
                + proto_dict: dict[relation, tensor.Tensor([1 x encoder_output_dim])]
                + concentration: dict[relation, tensor.Tensor([1 x len(seen_relations)])]
            return:
                + no_name_loss = \lambda_1 * intra_class_loss +\lambda_2 * protoNCE_loss : torch.tensor()
        
        """

        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()
        new_protos_raw = protos_raw.copy()
        new_proto_dict = proto_dict.copy()
        optimizer = self.get_optimizer(args, encoder)

        def train_data(data_loader_, name = "", is_mem = False):
            intra_class_losses = []
            protoNCE_losses = []
            td = tqdm(data_loader_, desc=name)

            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                hiddens, reps = encoder.bert_forward(tokens)
                
                for relation in current_relations: 
#                      update_protos(self, protos_raw, protos_index, protos_hidden_len, hiddens, ind, labels, relation):
                    update_proto, relation_idx = self.update_protos(protos_raw, protos_index, protos_hidden_len, hiddens, ind, labels, relation)
                    new_protos_raw[relation_idx] = update_proto
                    new_proto_dict[self.rel2id[relation]] = update_proto
            
                #  ----------------------start intra class loss --------------------------
                # start map prototype for each sample in batch 
                balance_class = np.array(labels.to('cpu'))
                balance_class = [1/np.sum(balance_class == c) for c in balance_class]
                balance_class = torch.Tensor(balance_class) 
                # balance_class : 1 x batch_size 
                proto_batch = [new_proto_dict[(int)(rel.item())] for rel in labels]
                proto_batch = torch.cat([x.to(args.device) for x in proto_batch], dim = 0)
                # [batch_size x args.encoder_bert_out]
                # proto_batch: batch_size x encoder_output_dim
                # end map prototype for each sample in batch 
                # start calculate intra class loss
                euclid_distance = (proto_batch - hiddens)**2
                euclid_distance = torch.mean(euclid_distance, dim = -1) # 
                euclid_distance = euclid_distance.to(args.device)
                balance_class = balance_class.to(args.device)
                intra_class_loss = euclid_distance@balance_class
                intra_class_loss = intra_class_loss.view(-1)
                intra_class_loss = intra_class_loss * 0.01
                # end calculate intra class loss                 
          
                # add intra class loss into final_loss
                intra_class_losses.append(intra_class_loss.item())    
                # -----------------------start protoNCE class loss ----------------------------
                sup_contrastive_loss = self.moment.loss(reps, labels)
                protoNCE_loss = self.get_protoNCE_loss(args, hiddens, new_protos_raw, balance_class, seen_relations, labels, concentration)
                protoNCE_losses.append(protoNCE_loss.item())

                loss = sup_contrastive_loss + protoNCE_loss + intra_class_loss
                loss.backward(retain_graph = True)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                
                
                
                td.set_postfix(protoNCE_loss = np.array(protoNCE_losses).mean(), intra_class_loss=" {} ".format(np.array(intra_class_losses).mean()))

#                 no_name_loss.backward(retain_graph = True)
#                 torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
#                 optimizer.step()
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach(), hiddens.detach())
                for idx, relation in enumerate(seen_relations):
                    proto, len_hidden = self.get_proto_raw(self.moment.hiddens, self.moment.labels, relation)
                    protos_raw[idx] = proto
                    proto_dict[self.rel2id[relation]] = proto
                # đang xét đến trường hợp dữ liệu train chỉ có dữ liệu task hiệu tại không thêm mem vào
                # không update prototype quá khứ được. cũng tốt, tức là coi prototype quá khứ không đổi => không bị kiểu domain shift nhưng mà khi học task mới, encoder đẫ đổi?
                # prototype không bị domain shift nhưng emcoder bị domain shift không? 
                # trả lời: không, vì ta đã cố định những domain cũ, tức 
        for epoch_i in range(1):
            train_data(data_loader, "no_name_train_{}".format(epoch_i), is_mem=False)
            
            

    def episodic_distillation(args, encoder):
        """
            return:
                + episodic_loss: torch.tensor()
        """
        pass
    
    
#     def get_protoNCE_loss(self, args, hiddens, protos_raw, balance_class, seen_relations, labels, concentraion):
#         """
#             inputs: 
#                 + hiddens: torch.Tensor(requires_grad = True) : batch_size x 768
#                 + protos_raw: torch.Tensor(requires_grad = True) : n_seen_relations x 768 
#                 + balance_class: torch.Tenor(): [1 x batch_size]
#                 + seen_relations: list[str]
#                 + labels: torch.Tensor(): 1 x len(current_relations)
#                 + concentration : dict[relation; torch.Tensor([1 x len(seen_relations)])]
#             outputs:
#                 + protoNCE_loss: torch.tensor(requires_grad = True)
#         """
#         phi = []
#         hiddens = F.normalize(hiddens, p = 2, dim = 1)
#         protos_raw = torch.stack(protos_raw, dim =0)
#         protos_raw = F.normalize(protos_raw, p = 2, dim = 1)
        
#         for rel in labels:
#             phi.append(concentraion[(int)(rel.item())])
#         phi = torch.stack(phi, dim = 0)
#         # step 1: calculation contrastive loss between hiddens and protos_raw
#         protos_raw = protos_raw.to(args.device)
#         dot_product = torch.mm(hiddens, protos_raw.T)
#         # expect: dot_product : batch_size x n_seen_relations 
#         dot_product *= phi
#         dot_product = torch.exp(dot_product - torch.max(dot_product, dim = 1, keepdim = True)[0].detach()) + 1e-5
#         # labels = labels.to(args.device)
#         seen_relations = torch.Tensor([self.rel2id[i] for i in seen_relations]).to(args.device)
#         mask = labels.unsqueeze(1).repeat(1, seen_relations.shape[0]) == seen_relations
#         prob = -torch.log(dot_product / torch.sum(dot_product, dim = 1, keepdim = True))
#         contrastive_loss = torch.sum(prob*mask, dim = 1) * balance_class
#         contrastive_loss = torch.mean(contrastive_loss)

#         return contrastive_loss    
    def get_protoNCE_mem_loss(self, args, hiddens, protos_raw, balance_class, seen_relations, labels):
        """
            inputs: 
                + hiddens: torch.Tensor(requires_grad = True) : batch_size x 768
                + protos_raw: torch.Tensor(requires_grad = True) : n_seen_relations x 768 
                + balance_class: torch.Tenor(): [1 x batch_size]
                + seen_relations: list[str]
                + labels: torch.Tensor(): 1 x len(current_relations)
                + concentration : dict[relation; torch.Tensor([1 x len(seen_relations)])]
            outputs:
                + protoNCE_loss: torch.tensor(requires_grad = True)
        """
        hiddens = F.normalize(hiddens, p = 2, dim = 1)
        protos_raw = torch.stack(protos_raw, dim =0)
        protos_raw = F.normalize(protos_raw, p = 2, dim = 1)
        # step 1: calculation contrastive loss between hiddens and protos_raw
        protos_raw = protos_raw.to(args.device)
        dot_product = torch.mm(hiddens, protos_raw.T)
        # expect: dot_product : batch_size x n_seen_relations 
        dot_product = torch.exp(dot_product - torch.max(dot_product, dim = 1, keepdim = True)[0].detach()) + 1e-5
        labels = labels.to(args.device)
        seen_relations = torch.Tensor([self.rel2id[i] for i in seen_relations]).to(args.device)
        mask = labels.unsqueeze(1).repeat(1, seen_relations.shape[0]) == seen_relations
        
        prob = - torch.log(dot_product / torch.sum(dot_product, dim = 1, keepdim = True))
        contrastive_loss = torch.sum(prob*mask, dim = 1) * balance_class
        contrastive_loss = torch.mean(contrastive_loss)

        return contrastive_loss
    def train_mem_model(self, args, encoder, mem_data, proto_mem, epochs, seen_relations, protos_raw):
        history_nums = len(seen_relations) - args.rel_per_task
        
        if len(proto_mem)>0:
            
            proto_mem = F.normalize(torch.cat(proto_mem, dim = 0), p =2, dim=1)
            dist = dot_dist(proto_mem, proto_mem)
            dist = dist.to(args.device)

        mem_loader = get_data_loader(args, mem_data, shuffle=True)
        encoder.train()
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        optimizer = self.get_optimizer(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            kl_losses = []
            protoNCE_losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):

                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                balance_class = np.array(labels)
                balance_class = [1/np.sum(balance_class == c) for c in balance_class]
                balance_class = torch.Tensor(balance_class)
                balance_class = balance_class.to(args.device)
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                zz, reps = encoder.bert_forward(tokens)
                hidden = reps


                need_ratio_compute = ind < history_nums * args.num_protos
                total_need = need_ratio_compute.sum()
                
                if total_need >0 :
                    # Knowledge Distillation for Relieve Forgetting
                    need_ind = ind[need_ratio_compute]
                    need_labels = labels[need_ratio_compute]
                    temp_labels = [map_relid2tempid[x.item()] for x in need_labels]
                    gold_dist = dist[temp_labels]
                    current_proto = self.moment.get_mem_proto()[:history_nums]
                    this_dist = dot_dist(hidden[need_ratio_compute], current_proto.to(args.device))
                    loss1 = self.kl_div_loss(gold_dist, this_dist, t=args.kl_temp)
                    loss1.backward(retain_graph=True)
                else:
                    loss1 = 0.0

                #  Contrastive Replay
                cl_loss = self.moment.loss(reps, labels, is_mem=True, mapping=map_relid2tempid)

                if isinstance(loss1, float):
                    kl_losses.append(loss1)
                else:
                    kl_losses.append(loss1.item())
                loss = cl_loss
#                     def get_protoNCE_mem_loss(self, args, hiddens, protos_raw, balance_class, seen_relations, label):

                protoNCE_loss = self.get_protoNCE_mem_loss(args, zz, protos_raw, balance_class, seen_relations, labels)
                protoNCE_losses.append(protoNCE_loss.item())
                final_loss = cl_loss + protoNCE_loss
                if isinstance(loss, float):
                    losses.append(loss)
                    td.set_postfix(loss = np.array(losses).mean(),  kl_loss = np.array(kl_losses).mean(), protoNCE_loss = np.array(protoNCE_losses).mean())
                    # update moemnt
                    if is_mem:
                        self.moment.update_mem(ind, reps.detach(), hidden.detach())
                    else:
                        self.moment.update(ind, reps.detach(), zz.detach())
                    continue
                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean(),  kl_loss = np.array(kl_losses).mean(), protoNCE_loss = np.array(protoNCE_losses).mean())
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach(), zz.detach())
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(mem_loader, "memory_train_{}".format(epoch_i), is_mem=True)
    def kl_div_loss(self, x1, x2, t=10):

        batch_dist = F.softmax(t * x1, dim=1)
        temp_dist = F.log_softmax(t * x2, dim=1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
        return loss

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, test_data, protos4eval, featrues4eval, seen_relations):
        data_loader = get_data_loader(args, test_data, batch_size=1)
        encoder.eval()
        n = len(test_data)
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        correct = 0
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            labels = labels.to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            hidden, reps = encoder.bert_forward(tokens)
            labels = [map_relid2tempid[x.item()] for x in labels]
            logits = -osdist(hidden, protos4eval)
            seen_relation_ids = [self.rel2id[relation] for relation in seen_relations]
            seen_relation_ids = [map_relid2tempid[x] for x in seen_relation_ids]
            seen_sim = logits[:,seen_relation_ids]
            seen_sim = seen_sim.cpu().data.numpy()
            max_smi = np.max(seen_sim,axis=1)
            label_smi = logits[:,labels].cpu().data.numpy()
            if label_smi >= max_smi:
                correct += 1
        return correct/n
    
    def get_proto_raw(self, hiddens, labels, relation):
        """
        input: 
            + hiddens: torch.Tensor() : datalen x args.encoder_output_size : get from self.momment.hiddens
            + labels : list[int] : label corresponse each hidden
            + relation : [str] relation in current task
        output: 
            + proto: torch.tensor() : 1 x args.encoder_output_size: prototype of this relation
            + len_hidden: torch.tensor(int): number of hidden in hiddens has label is relation          
        """
        mask = torch.Tensor(np.array(labels.to('cpu')) == self.rel2id[relation]).cuda()
        mask = mask.reshape(1,-1)
        hiddens_of_relation = torch.matmul(mask, hiddens)
        proto = torch.mean(hiddens_of_relation, dim = 0)
        proto = proto.reshape(1, -1)
        proto = proto.to('cuda:0')
        len_hidden = torch.sum(mask)
        return proto, len_hidden
        
        
    
    def train(self, args):
        # set training batch
        for i in range(args.total_round):
            test_cur = []
            test_total = []
            # set random seed
            random.seed(args.seed+i*100)

            # sampler setup
            sampler = data_sampler(args=args, seed=args.seed+i*100)
            self.id2rel = sampler.id2rel
            self.rel2id = sampler.rel2id
            # encoder setup
            encoder = Encoder(args=args).to(args.device)

            # initialize memory and prototypes
            num_class = len(sampler.id2rel)
            memorized_samples = {}

            # load data and start computation
            
            history_relation = []
            proto4repaly = []
            protos_raw = []
            x_protos_raw = []
            protos_dict = {}
            protos_hidden_len = []
            protos_index = []
            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):

                print(current_relations)
                # Initial
                train_data_for_initial = []
                for relation in current_relations:
                    history_relation.append(relation)
                    train_data_for_initial += training_data[relation]
                # train model
                # no memory. first train with current task
                self.moment = Moment(args)
                self.moment.init_moment(args, encoder, train_data_for_initial, is_memory=False)
#                 self.train_simple_model(args, encoder, train_data_for_initial, args.step1_epochs)
                # select current task sample
                for idx, relation in enumerate(seen_relations):
                    proto, len_hidden = self.get_proto_raw(self.moment.hiddens, self.moment.labels, relation)
                    if relation in current_relations:
                        protos_raw.append(proto)
                        protos_dict[self.rel2id[relation]] = proto
                        protos_hidden_len.append(len_hidden)
                        protos_index.append(self.rel2id[relation])
                    else:
                        protos_raw[idx] = proto
                        protos_dict[self.rel2id[relation]] = proto
                concentration = self.get_concentration(args, encoder, training_data, protos_raw, current_relations)
                self.train_no_name_model(args,encoder, train_data_for_initial, protos_raw, seen_relations, current_relations, protos_dict,protos_index, protos_hidden_len, concentration)
                for idx, relation in enumerate(seen_relations):
                    proto, len_hidden = self.get_proto_raw(self.moment.hiddens, self.moment.labels, relation)
                    protos_raw[idx] = proto
                    protos_dict[self.rel2id[relation]] = proto
                # repaly
                x_protos_raw = protos_raw[:-8].copy() 
                if len(memorized_samples)>0:
                    # select current task sample
                    for relation in current_relations:
                        memorized_samples[relation], _, proto_raw = self.select_data(args, encoder, training_data[relation])
                        x_protos_raw.append(proto_raw)
                    train_data_for_memory = []
                    for relation in history_relation:
                        train_data_for_memory += memorized_samples[relation]
                    
                    self.moment.init_moment(args, encoder, train_data_for_memory, is_memory=True)
                    self.train_mem_model(args, encoder, train_data_for_memory, x_protos_raw, args.step2_epochs, seen_relations, protos_raw)

                feat_mem = []
                proto_mem = []

                for relation in current_relations:
                    memorized_samples[relation], feat, temp_proto = self.select_data(args, encoder, training_data[relation])
                    feat_mem.append(feat)
                    proto_mem.append(temp_proto)

                feat_mem = torch.cat(feat_mem, dim=0)
                temp_proto = torch.stack(proto_mem, dim=0)

                protos4eval = []
                featrues4eval = []
                self.lbs = []
                for relation in history_relation:
                    if relation not in current_relations:
                        
                        protos, featrues = self.get_proto(args, encoder, memorized_samples[relation])
                        protos4eval.append(protos)
                        featrues4eval.append(featrues)
                
                if protos4eval:
                    
                    protos4eval = torch.cat(protos4eval, dim=0).detach()
                    protos4eval = torch.cat([protos4eval, temp_proto.to(args.device)], dim=0)

                else:
                    protos4eval = temp_proto.to(args.device)
                proto4repaly = protos4eval.clone()

                test_data_1 = []
                for relation in current_relations:
                    test_data_1 += test_data[relation]

                test_data_2 = []
                for relation in seen_relations:
                    test_data_2 += historic_test_data[relation]

                cur_acc = self.evaluate_strict_model(args, encoder, test_data_1, protos4eval, featrues4eval,seen_relations)
                total_acc = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval, featrues4eval,seen_relations)

                print(f'Restart Num {i+1}')
                print(f'task--{steps + 1}:')
                print(f'current test acc:{cur_acc}')
                print(f'history test acc:{total_acc}')
                test_cur.append(cur_acc)
                test_total.append(total_acc)
                
                print(test_cur)
                print(test_total)
                del self.moment
