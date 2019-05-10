from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import random
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from data import get_id_relation, tokenize_relation, build_vocab
from collections import OrderedDict
from attention import SimpleEncoder
import math

#from yellowfin import YFOptimizer
class Packed(nn.Module):
    '''
    usage:
    initialize your LSTM as lstm = Packed(nn.LSTM(...))
    '''

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    @property
    def batch_first(self):
        return self.rnn.batch_first

    def forward(self, inputs, lengths, hidden=None, max_length=None):
        lengths = torch.tensor(lengths)
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices]
        outputs, (h, c) = self.rnn(nn.utils.rnn.pack_padded_sequence(inputs, lens.tolist(), batch_first=self.batch_first), hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first, total_length=max_length)
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]
        return outputs, (h, c)

class AttnEncoder(nn.Module):
    def __init__(self, d_hid):
        super(AttnEncoder, self).__init__()
        self.attn_linear = nn.Linear(d_hid, 1, bias=False)

    def forward(self, x, x_mask):
        x_attn = self.attn_linear(x)
        x_attn = x_attn - (1 - x_mask.unsqueeze(2))*1e8
        x_attn = F.softmax(x_attn, dim=1)
        return (x*x_attn).sum(1)

def gen_mask_based_length(lengths, cuda_id=0):
    batch_size = len(lengths)
    doc_size = max(lengths)
    masks = torch.ones(batch_size, doc_size)
    index_matrix = torch.arange(0, doc_size).expand(batch_size, -1)
    index_matrix = index_matrix.long()
    doc_lengths = torch.tensor(lengths).cpu().view(-1,1)
    doc_lengths_matrix = doc_lengths.expand(-1, doc_size)
    masks[torch.ge(index_matrix-doc_lengths_matrix, 0)] = 0
    return masks.cuda(cuda_id)

class LSTM(nn.Module):
    def __init__(self, arg, vocab_embedding):
        super(LSTM, self).__init__()
        self.hidden_dim = arg['hidden_size']
        vocab_size, embedding_dim = vocab_embedding.shape
        self.pooler = AttnEncoder(arg['hidden_size'])

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(vocab_embedding))
        self.lstm = Packed(nn.LSTM(embedding_dim, self.hidden_dim,
                                   bidirectional=False))

    def forward(self, padded_sentences, lengths, cuda_id=0):
        padded_embeds = self.embedding(padded_sentences)
        #print(len(padded_sentences))
        lstm_out, hidden_state = self.lstm(padded_embeds, lengths)
        lstm_out = lstm_out.permute([1,0,2])
        mask = gen_mask_based_length(lengths, cuda_id)
        return self.pooler(lstm_out, mask)
        #permuted_hidden = hidden_state[0].permute([1,0,2]).contiguous()
        #return permuted_hidden.view(permuted_hidden.size(0), -1)

class Agent(nn.Module):
    """The agent class, it includes model definition and forward functions"""
    def __init__(self, arg,cuda_id=0, use_sgd=False):
        """
        Parameters:
        embed_size --- size of relation/entity embeddings
        rollout_train --- width of search during training
        rollout_test --- width od search during inference
        num_relation ---- num_of_relations
        num_entity --- num of entities
        policy_layers --- num of RNN layers
        hidden_size --- dimension of RNN hidden layer

        """
        super(Agent, self).__init__()
        self.record_path = [[],[]]
        self.use_path_encoder = True
        self.cuda_id = cuda_id
        for k, v in arg.items(): setattr(self, k, v)
        self.batch_size *= self.num_rollouts
        self.num_relation = len(self.relation_vocab)
        self.num_entity = len(self.entity_vocab)
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']

        # relation embedding matrix
        self.relation_emb = nn.Embedding(self.num_relation, self.embed_size)
        # nn.init.xavier_uniform(self.relation_emb.weight)

        self.id_rels = get_id_relation(arg)
        self.token_vocab = build_vocab(arg)
        token_embed = self.token_vocab.embedding.idx_to_vec
        #self.relation_enc = LSTM(arg, token_embed.asnumpy())
        self.all_relation_tokens, self.all_relation_token_lengths =\
            self.all_tokenized_relations()

        # entity embedding matrix
        self.entity_emb = nn.Embedding(self.num_entity, self.embed_size)
        # nn.init.xavier_uniform(self.entity_emb.weight)

        # recurrent policy
        self.rnns = []
        for i in range(self.policy_layers):
            if i == 0:
                self.rnns.append(nn.LSTMCell(2*self.embed_size, self.hidden_size).cuda(self.cuda_id))
            else:
                self.rnns.append(nn.LSTMCell(self.hidden_size, self.hidden_size).cuda(self.cuda_id))
        # self.hidden_1 = nn.Linear(self.hidden_size + 2*self.embed_size, 4*self.hidden_size)
        self.path_encoder = nn.LSTM(self.embed_size, self.embed_size, batch_first=True)
        #self.path_encoder = SimpleEncoder(self.embed_size, 2, 2)
        #self.path_encoder = None
        self.surrogate_path = {}
        self.seen_query_rels = []
        self.surrogate_path_limit = 256
        self.hidden_1 = nn.Linear(self.hidden_size + 3*self.embed_size, 4*self.hidden_size)
        self.hidden_2 = nn.Linear(4*self.hidden_size, 2*self.embed_size)

        self.update_steps = 0
        if not use_sgd:
            #self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)
            self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optim = optim.SGD(self.parameters(), lr=self.learning_rate)

        # self.optim = YFOptimizer(self.parameters(), lr=self.learning_rate)

        self.baseline = 0.0

    def forward(self, next_rels, next_ents, pre_states, pre_rels, query_rels, curr_ents, params=None):
        """
        batch_size here equals to ortiginal batch_size * num_rollouts
        next_rels, next_ents ---- batch * action_num
        pre_states --- a list of previous RNN states
        query_rels ---

        """
        #if params is None:
        #    params = OrderedDict(self.named_parameters())

        next_rel_emb = self.relation_emb(next_rels)
        next_ent_emb = self.entity_emb(next_ents)
        action_emb = torch.cat((next_rel_emb, next_ent_emb), dim=2) # batch_size * action_num * 2d

        query_rel_emb = self.query_relation_emb(query_rels)
        curr_ent_emb = self.entity_emb(curr_ents)
        pre_rel_emb = self.relation_emb(pre_rels)

        # update RNN states
        inputs = torch.cat((pre_rel_emb, curr_ent_emb), dim=1)
        next_states = []
        for i in range(self.policy_layers):
            h_, c_ = self.rnns[i](inputs, pre_states[i])
            next_states.append((h_,c_))
            inputs = h_

        # outputs scores for each action
        # states_for_pred = torch.cat((h_, curr_ent_emb.squeeze(), query_rel_emb.squeeze()), dim=1)
        states_for_pred = torch.cat((h_, pre_rel_emb.squeeze(), curr_ent_emb.squeeze(), query_rel_emb.squeeze()), dim=1)
        #print(states_for_pred.size(), query_rel_emb.size())
        #print(action_emb.size())
        logits = torch.bmm(action_emb, F.relu(self.hidden_2(F.relu(self.hidden_1(states_for_pred)))).unsqueeze(2)).squeeze() # batch * action_num
        padded = (torch.ones(next_rels.size()) * self.rPAD).cuda(self.cuda_id)
        padded_actions = torch.eq(padded, next_rels.data.float())
        logits[padded_actions] = -99999.0
        logits = F.softmax(logits)

        return logits, next_states

    def all_tokenized_relations(self):
        relation_ids = list(self.id_rels.keys())
        relation_names = [self.id_rels[this_id] for this_id in relation_ids]
        token_ids = [torch.tensor(self.token_vocab(tokenize_relation(rel)))
                     for rel in relation_names]
        token_lengths = torch.tensor([len(_) for _ in token_ids])
        token_ids = pad_sequence(token_ids)
        #token_ids = self.token_vocab[tokenize_names]
        return token_ids.cuda(self.cuda_id), token_lengths.cuda(self.cuda_id)

    def query_relation_emb(self, relation_ids):
        query_rel = relation_ids[0]
        rel_id =  int(query_rel)
        if len(self.surrogate_path[rel_id]) > 0:
            record_actions = torch.from_numpy(self.surrogate_path[rel_id]).long().cuda()
            #sel_idx = np.random.choice(list(range(len(record_actions))), 5)
            sel_idx = list(range(len(record_actions)))
            record_actions = record_actions[sel_idx]
            record_action_embed = self.relation_emb(record_actions)
            #record_action_embed = self.relation_emb(record_actions)
            output, (h, c) = self.path_encoder(record_action_embed)
            #h = self.path_encoder(record_action_embed)
            h = h.view(len(record_actions), -1)
            query_rel_embed = torch.mean(h, 0)
        else:
            return self.relation_emb(relation_ids)
            query_rel_embed = self.relation_emb(query_rel)
        #print(query_rel_embed.size())
        return query_rel_embed.view(1,-1).expand(len(relation_ids), -1).contiguous()


    def _relation_emb(self, relation_ids):
        #print(relation_ids.size())
        #print(self.all_relation_tokens.size(), self.all_relation_token_lengths.size())
        all_relation_embeddings = self.relation_enc(self.all_relation_tokens,
                                                    self.all_relation_token_lengths,
                                                    self.cuda_id)
        set_size = list(relation_ids.size())
        set_size.append(-1)
        ret_rel_emb = all_relation_embeddings[relation_ids.view(-1)]
        #print(all_relation_embeddings.size(), set_size)
        return ret_rel_emb.view(set_size)

    def init_rnn_states(self, batch_size):
        init = []
        for i in range(self.policy_layers):
            init.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda(self.cuda_id)), Variable(torch.zeros(batch_size, self.hidden_size).cuda(self.cuda_id))))
        return init

    def decay_lr(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] = max(self.alpha2, param_group['lr']*0.01)

    def update_path_embed(self, rewards, record_path_rel, args=None, is_lstm=False, reasoner=None, not_updated = True, update_embed=False, acc_path=False):
        if reasoner is None:
            reasoner = self.path_encoder
        record_actions, query_rels = record_path_rel
        record_actions = torch.stack(record_actions, 1)
        #print(query_rels)
        reward_t = torch.from_numpy(rewards)
        sel_path_idx = reward_t == 1
        record_actions = record_actions[sel_path_idx]
        query_rels = query_rels[sel_path_idx]
        embed_dict = {}
        if acc_path and torch.sum(sel_path_idx) > 0:
            query_rel_id = int(query_rels[0])
            self.surrogate_path[query_rel_id] = record_actions if query_rel_id not in self.surrogate_path\
                    else torch.cat((self.surrogate_path[query_rel_id], record_actions), 0)
            self.surrogate_path[query_rel_id] = self.surrogate_path[query_rel_id][-self.surrogate_path_limit:]
        if update_embed and torch.sum(sel_path_idx) > 0:
            query_rel_id = int(query_rels[0])
            record_actions = self.surrogate_path[query_rel_id]
            record_action_embed = self.relation_emb(record_actions)
            #query_relation_embed = self.relation_emb(query_rels)
            query_relation_embed = self.query_relation_emb(query_rels)
            if is_lstm:
                output, (h, c) = reasoner(record_action_embed)
            else:
                h = reasoner(record_action_embed)
            relation_embed_t = self.relation_emb.weight
            relation_embed_t.data[query_rel_id] = torch.mean(h, 0)
            '''
            path_embed =  h.view(query_relation_embed.size())
            for i in range(len(query_rels)):
                if query_rels[i] in embed_dict:
                    embed_dict[query_rels[i]].append(path_embed[i])
                else:
                    embed_dict[query_rels[i]] = [path_embed[i]]
            for rel in embed_dict:
                if not_updated:
                    relation_embed_t.data[rel] = torch.mean(torch.stack(embed_dict[rel]), 0)
                else:
                    relation_embed_t.data[rel] = relation_embed_t.data[rel] * 0.9 + 0.1*torch.mean(torch.stack(embed_dict[rel]), 0)
            '''
            self.surrogate_path = []
            return False
        return True

    def update(self, rewards, record_action_probs, record_probs, record_path_rel, decay_lr=False, args=None, reasoner_only=False):
        # discounted rewards
        if args['new_reward']:
            discounted_rewards = rewards.transpose()
        else:
            discounted_rewards = np.zeros((rewards.shape[0], self.path_length))
            discounted_rewards[:,-1] = rewards

        record_actions, query_rels = record_path_rel
        record_actions = torch.stack(record_actions, 1)
        embed_loss = None
        reward_t = torch.from_numpy(rewards)
        sel_path_idx = reward_t == 1
        #print(reward_t, sel_path_idx)
        record_actions = record_actions[sel_path_idx]
        query_rels = query_rels[sel_path_idx]
        #self.record_path[0].append(record_actions)
        #self.record_path[1].append(query_rels)
        if False and self.use_path_encoder and torch.sum(sel_path_idx) > 0:
            query_rel_id = int(query_rels[0])
            self.surrogate_path[query_rel_id] = record_actions if query_rel_id not in self.surrogate_path\
                    else torch.cat((self.surrogate_path[query_rel_id], record_actions), 0)
            self.surrogate_path[query_rel_id] = self.surrogate_path[query_rel_id][-self.surrogate_path_limit:]
            '''
            record_action_embed = self.relation_emb(record_actions)
            query_relation_embed = self.query_relation_emb(query_rels).detach()
            #output, (h, c) = self.path_encoder(record_action_embed)
            h = self.path_encoder(record_action_embed)
            dis = nn.PairwiseDistance(p=2)
            #dis = nn.CosineSimilarity(dim=1)
            h = h.view(query_relation_embed.size())
            #print(h.size())
            #print(query_relation_embed.size())
            embed_loss = torch.mean(dis(h, query_relation_embed))
            '''

        #discounted_rewards = rewards.transpose()
        for i in range(1, self.path_length):
            discounted_rewards[:, -1-i] = discounted_rewards[:, -1-i] + self.gamma * discounted_rewards[:, -1-i+1]
        final_reward = discounted_rewards - self.baseline
        reward_mean = np.mean(final_reward)
        reward_var = np.var(final_reward)
        reward_std = np.sqrt(reward_var) + 1e-6
        final_reward = (final_reward - reward_mean) / reward_std

        # beta = self.beta * 0.9 ** (self.update_steps // 400)
        beta = self.beta
        # beta = 0.0
        self.update_steps += 1

        # entropy loss
        record_probs = torch.stack(record_probs, dim=2)
        p_logp = torch.log(record_probs + 1e-8) * record_probs
        self.entropy_loss = beta * torch.mean(torch.sum(p_logp, dim=1))

        # RL loss
        record_action_probs = torch.stack(record_action_probs, dim=1)
        self.rl_loss = - torch.mean(torch.log(record_action_probs + 1e-8) * Variable(torch.FloatTensor(final_reward)).cuda(self.cuda_id))


        self.baseline = self.Lambda * np.mean(discounted_rewards) + (1-self.Lambda) * self.baseline
        self.loss = self.entropy_loss + self.rl_loss
        if False and embed_loss is not None:
            self.loss += 0.1*embed_loss
            print(embed_loss.data)
        if reasoner_only:
            reasoner_optim = optim.SGD(self.path_encoder.parameters(), lr=self.learning_rate)
            reasoner_optim.zero_grad()
            self.loss.backward()
            reasoner_optim.step()
        else:
            self.optim.zero_grad()
            self.loss.backward()
            nn.utils.clip_grad_norm(self.parameters(), self.grad_clip_norm)
            self.optim.step()
        if decay_lr:
            self.decay_lr()

        # # decrease the learning rate here
        # if self.update_steps % 500 == 0 and self.update_steps > 0:
        #     self.learning_rate = 0.9 * self.learning_rate
        #     self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)
            # self.optim = YFOptimizer(self.parameters(), lr=self.learning_rate)

        return self.loss.item(), np.mean(rewards)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def save_record_path(self, path):
        record_actions = torch.cat(self.record_path[0], 0)
        query_rels = torch.cat(self.record_path[1], 0)
        torch.save(record_actions, path+'record_actions')
        torch.save(query_rels, path+'query_rels')

    def load_record_path(self, path):
        record_actions = torch.load(path+'record_actions')
        query_rels = torch.load(path+'query_rels')
        self.record_path = [[record_actions], [query_rels]]

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_loss(self, rewards, record_action_probs, record_probs, args,
            record_path_rel):
        # discounted rewards

        if args['new_reward']:
            discounted_rewards = rewards.transpose()
        else:
            discounted_rewards = np.zeros((rewards.shape[0], self.path_length))
            discounted_rewards[:,-1] = rewards

        record_actions, query_rels = record_path_rel
        record_actions = torch.stack(record_actions, 1)
        query_rel_id = int(query_rels[0])
        if query_rel_id not in self.seen_query_rels:
            self.seen_query_rels.append(query_rel_id)
        embed_loss = None
        reward_t = torch.from_numpy(rewards)
        sel_path_idx = reward_t == 1
        #print(reward_t, sel_path_idx)
        #print(query_rels.size())
        record_actions = record_actions[sel_path_idx]
        query_rels = query_rels[sel_path_idx]
        #self.record_path[0].append(record_actions)
        #self.record_path[1].append(query_rels)
        if False and self.use_path_encoder and torch.sum(sel_path_idx) > 0:
            query_rel_id = int(query_rels[0])
            self.surrogate_path[query_rel_id] = record_actions if query_rel_id not in self.surrogate_path\
                    else torch.cat((self.surrogate_path[query_rel_id], record_actions), 0)
            self.surrogate_path[query_rel_id] = self.surrogate_path[query_rel_id][-self.surrogate_path_limit:]
            '''
            record_action_embed = self.relation_emb(record_actions)
            query_relation_embed = self.relation_emb(query_rels).detach()
            #output, (h, c) = self.path_encoder(record_action_embed)
            h = self.path_encoder(record_action_embed)
            dis = nn.PairwiseDistance(p=2)
            #dis = nn.CosineSimilarity(dim=1)
            h = h.view(query_relation_embed.size())
            #print(h.size())
            #print(query_relation_embed.size())
            embed_loss = torch.mean(dis(h, query_relation_embed))
            '''

        for i in range(1, self.path_length):
            discounted_rewards[:, -1-i] = discounted_rewards[:, -1-i] + self.gamma * discounted_rewards[:, -1-i+1]
        final_reward = discounted_rewards - self.baseline
        reward_mean = np.mean(final_reward)
        reward_var = np.var(final_reward)
        reward_std = np.sqrt(reward_var) + 1e-6
        final_reward = (final_reward - reward_mean) / reward_std

        # beta = self.beta * 0.9 ** (self.update_steps // 400)
        beta = self.beta
        # beta = 0.0
        #self.update_steps += 1

        # entropy loss
        record_probs = torch.stack(record_probs, dim=2)
        p_logp = torch.log(record_probs + 1e-8) * record_probs
        self.entropy_loss = beta * torch.mean(torch.sum(p_logp, dim=1))

        # RL loss
        record_action_probs = torch.stack(record_action_probs, dim=1)
        self.rl_loss = - torch.mean(torch.log(record_action_probs + 1e-8) * Variable(torch.FloatTensor(final_reward)).cuda(self.cuda_id))


        self.baseline = self.Lambda * np.mean(discounted_rewards) + (1-self.Lambda) * self.baseline
        self.loss = self.entropy_loss + self.rl_loss
        if embed_loss is not None:
            self.loss += 0.1*embed_loss
            print(embed_loss.data)
        #print(self.loss, embed_loss)
        return self.loss
        #return torch.autograd.grad(self.loss, self.parameters)
        #self.optim.zero_grad()
        #self.loss.backward()
        #nn.utils.clip_grad_norm(self.parameters(), self.grad_clip_norm)
        #self.optim.step()

        # # decrease the learning rate here
        # if self.update_steps % 500 == 0 and self.update_steps > 0:
        #     self.learning_rate = 0.9 * self.learning_rate
        #     self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)
            # self.optim = YFOptimizer(self.parameters(), lr=self.learning_rate)

        #return self.loss.item(), np.mean(rewards)
    def update_params(self, loss, step_size=0.5, only_path_encoder=False):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        #grads = torch.autograd.grad(loss, self.parameters(),
        #    create_graph=not first_order)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.grad_clip_norm)
        self.optim.step()
        '''
        updated_params = self.path_encoder.state_dict() if only_path_encoder else self.state_dict()
        #for (name, param), grad in zip(self.named_parameters(), grads):
        grad_params = self.path_encoder.named_parameters() if only_path_encoder else self.named_parameters()
        for (name, param) in grad_params:
            updated_params[name] = param.clone()
            if param.grad is not None:
                updated_params[name] -= step_size * param.grad

        return updated_params
        '''

    def train_path_reasoner(self):
        record_actions = torch.cat(self.record_path[0], 0)
        query_rels = torch.cat(self.record_path[1], 0)
        optim = torch.optim.Adam(self.path_encoder.parameters(), lr=self.path_lr)
        index = list(range(len(query_rels)))
        for epoch in range(self.path_epoch):
            record_actions = record_actions[index]
            query_rels = query_rels[index]
            random.shuffle(index)
            num_batches = math.ceil(len(record_actions)/self.path_batch_size)
            batch_size = self.path_batch_size
            for i in range(num_batches):
                batch_x = record_actions[i*batch_size:(i+1)*batch_size]
                batch_y = query_rels[i*batch_size:(i+1)*batch_size]
                x_embed = self.relation_emb(batch_x)
                y_embed = self.relation_emb(batch_y)
                #output, (h, c) = self.path_encoder(x_embed)
                h = self.paath_encoder(x_embed)
                path_enc = h.view(y_embed.size())
                dis = nn.PairwiseDistance(p=2)
                embed_loss = torch.mean(dis(path_enc, y_embed))
                print(embed_loss)
                optim.zero_grad()
                embed_loss.backward()
                optim.step()
    def train_given_reasoner(self, model, data_path):
        if model is None:
            model = self.path_encoder
            is_lstm = True
        else:
            #self.path_encoder = model
            is_lstm = False
        self.load_record_path(data_path)
        record_actions = torch.cat(self.record_path[0], 0)
        query_rels = torch.cat(self.record_path[1], 0)
        #print(len(record_actions))
        optim = torch.optim.Adam(model.parameters(), lr=self.path_lr)
        all_rels = torch.unique(query_rels)
        #print(all_rels[-5:])
        keep_samples_index = []
        for rel in all_rels:
            keep_samples_index.append((query_rels == rel).nonzero()[-100:])
        keep_samples_index = torch.cat(keep_samples_index, 0).flatten()
        #print(keep_samples_index[-5:])
        #print(record_actions.size(), query_rels.size())
        query_rels = query_rels.index_select(0,keep_samples_index)
        record_actions = record_actions.index_select(0,keep_samples_index)
        index = list(range(len(query_rels)))
        for epoch in range(self.path_epoch):
            record_actions = record_actions[index]
            query_rels = query_rels[index]
            random.shuffle(index)
            num_batches = math.ceil(len(record_actions)/self.path_batch_size)
            batch_size = self.path_batch_size
            for i in range(num_batches):
                batch_x = record_actions[i*batch_size:(i+1)*batch_size]
                batch_y = query_rels[i*batch_size:(i+1)*batch_size]
                x_embed = self.relation_emb(batch_x)
                y_embed = self.relation_emb(batch_y)
                if is_lstm:
                    output, (h, c) = model(x_embed)
                else:
                    h = model(x_embed)
                path_enc = h.view(y_embed.size())
                dis = nn.PairwiseDistance(p=2)
                embed_loss = torch.mean(dis(path_enc, y_embed))
                print(embed_loss)
                optim.zero_grad()
                embed_loss.backward()
                optim.step()
        return model
    def update_seen_query_embeds(self):
        for rel in self.seen_query_rels:
            rel_embed = self.query_relation_emb([torch.tensor(rel).cuda()])
            relation_embed_t = self.relation_emb.weight
            relation_embed_t.data[rel] = rel_embed
    def update_path_set(self, record_actions, query_id):
        record_actions = torch.stack(record_actions, 1)
        sel_idx = np.random.choice(len(record_actions), self.surrogate_path_limit)
        #self.surrogate_path[query_id] = record_actions[sel_idx]
        self.surrogate_path[query_id] = record_actions if query_id not in self.surrogate_path\
                    else torch.cat((self.surrogate_path[query_id], record_actions), 0)
        self.surrogate_path[query_id] = self.surrogate_path[query_id][-self.surrogate_path_limit:]
    def set_path_encoder(self):
        self.path_encoder = SimpleEncoder(self.embed_size, 2, 2)
