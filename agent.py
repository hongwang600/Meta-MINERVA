from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

#from yellowfin import YFOptimizer

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

    def init_rnn_states(self, batch_size):
        init = []
        for i in range(self.policy_layers):
            init.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()), Variable(torch.zeros(batch_size, self.hidden_size).cuda())))  
        return init

    def update(self, rewards, record_action_probs, record_probs):
        # discounted rewards

        discounted_rewards = np.zeros((rewards.shape[0], self.path_length))
        discounted_rewards[:,-1] = rewards
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










