from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
torch.manual_seed(1)
np.random.seed(1)
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from data import get_id_relation, tokenize_relation, build_vocab

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

def gen_mask_based_length(lengths):
    batch_size = len(lengths)
    doc_size = max(lengths)
    masks = torch.ones(batch_size, doc_size)
    index_matrix = torch.arange(0, doc_size).expand(batch_size, -1)
    index_matrix = index_matrix.long()
    doc_lengths = torch.tensor(lengths).cpu().view(-1,1)
    doc_lengths_matrix = doc_lengths.expand(-1, doc_size)
    masks[torch.ge(index_matrix-doc_lengths_matrix, 0)] = 0
    return masks.cuda()

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

    def forward(self, padded_sentences, lengths):
        padded_embeds = self.embedding(padded_sentences)
        #print(len(padded_sentences))
        lstm_out, hidden_state = self.lstm(padded_embeds, lengths)
        lstm_out = lstm_out.permute([1,0,2])
        mask = gen_mask_based_length(lengths)
        return self.pooler(lstm_out, mask)
        #permuted_hidden = hidden_state[0].permute([1,0,2]).contiguous()
        #return permuted_hidden.view(permuted_hidden.size(0), -1)

class Agent(nn.Module):
    """The agent class, it includes model definition and forward functions"""
    def __init__(self, arg):
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
                self.rnns.append(nn.LSTMCell(2*self.embed_size, self.hidden_size).cuda())
            else:
                self.rnns.append(nn.LSTMCell(self.hidden_size, self.hidden_size).cuda())
        # self.hidden_1 = nn.Linear(self.hidden_size + 2*self.embed_size, 4*self.hidden_size)
        self.path_encoder = SimpleEncoder(self.embed_size, 5, 5)
        self.hidden_1 = nn.Linear(self.hidden_size + 3*self.embed_size, 4*self.hidden_size)
        self.hidden_2 = nn.Linear(4*self.hidden_size, 2*self.embed_size)

        self.update_steps = 0
        self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)

        # self.optim = YFOptimizer(self.parameters(), lr=self.learning_rate)

        self.baseline = 0.0

    def forward(self, next_rels, next_ents, pre_states, pre_rels, query_rels, curr_ents):
        """
        batch_size here equals to ortiginal batch_size * num_rollouts
        next_rels, next_ents ---- batch * action_num
        pre_states --- a list of previous RNN states
        query_rels ---

        """

        next_rel_emb = self.relation_emb(next_rels)
        next_ent_emb = self.entity_emb(next_ents)
        action_emb = torch.cat((next_rel_emb, next_ent_emb), dim=2) # batch_size * action_num * 2d

        query_rel_emb = self.relation_emb(query_rels)
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
        logits = torch.bmm(action_emb, F.relu(self.hidden_2(F.relu(self.hidden_1(states_for_pred)))).unsqueeze(2)).squeeze() # batch * action_num
        padded = (torch.ones(next_rels.size()) * self.rPAD).cuda()
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
        return token_ids.cuda(), token_lengths.cuda()

    def _relation_emb(self, relation_ids):
        #print(relation_ids.size())
        #print(self.all_relation_tokens.size(), self.all_relation_token_lengths.size())
        all_relation_embeddings = self.relation_enc(self.all_relation_tokens,
                                                    self.all_relation_token_lengths)
        set_size = list(relation_ids.size())
        set_size.append(-1)
        ret_rel_emb = all_relation_embeddings[relation_ids.view(-1)]
        #print(all_relation_embeddings.size(), set_size)
        return ret_rel_emb.view(set_size)

    def init_rnn_states(self, batch_size):
        init = []
        for i in range(self.policy_layers):
            init.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()), Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        return init

    def update(self, rewards, record_action_probs, record_probs):
        # discounted rewards

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
        if self.use_path_encoder and torch.sum(sel_path_idx) > 0:
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
        self.rl_loss = - torch.mean(torch.log(record_action_probs + 1e-8) * Variable(torch.FloatTensor(final_reward)).cuda())


        self.baseline = self.Lambda * np.mean(discounted_rewards) + (1-self.Lambda) * self.baseline
        self.loss = self.entropy_loss + self.rl_loss

        if embed_loss is not None:
            self.loss += 0.1*embed_loss
            print(embed_loss.data)

        self.optim.zero_grad()
        self.loss.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.grad_clip_norm)
        self.optim.step()

        # # decrease the learning rate here
        # if self.update_steps % 500 == 0 and self.update_steps > 0:
        #     self.learning_rate = 0.9 * self.learning_rate
        #     self.optim = optim.Adam(self.parameters(), lr=self.learning_rate)
            # self.optim = YFOptimizer(self.parameters(), lr=self.learning_rate)

        return self.loss.item(), np.mean(rewards)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))










