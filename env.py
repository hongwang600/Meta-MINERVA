# from __future__ import absolute_import
# from __future__ import division

from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os
random.seed(1)
np.random.seed(1)


class RelationEntityBatcher():
    '''
    creates an indexed graph, store the answer entities for each query
    '''
    def __init__(self, input_dir, batch_size, entity_vocab, relation_vocab,
                 mode="train", batcher_triples=[]):
        self.input_dir = input_dir
        self.input_file = input_dir+'/{0}.txt'.format(mode)
        self.batch_size = batch_size
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.mode = mode
        self.create_triple_store(self.input_file, batcher_triples)
        self.data_size = min(1000, len(batcher_triples))
        # print("batcher loaded")


    def get_next_batch(self):
        if self.mode == 'train':
            yield self.yield_next_batch_train()
        else:
            yield self.yield_next_batch_test()


    def create_triple_store(self, input_file, batcher_triples):
        self.store_all_correct = defaultdict(set)
        self.store = []
        if self.mode == 'train':
            #with open(input_file) as raw_input_file:
            #    csv_file = csv.reader(raw_input_file, delimiter = '\t' )
            #    for line in csv_file:
            for line in batcher_triples:
                #print(line)
                e1 = line[0]
                r = line[1]
                e2 = line[2]
                if e1 in self.entity_vocab and e2 in self.entity_vocab:
                    e1 = self.entity_vocab[line[0]]
                    r = self.relation_vocab[line[1]]
                    e2 = self.entity_vocab[line[2]]
                    self.store.append([e1,r,e2])
                    self.store_all_correct[(e1, r)].add(e2)
            self.store = np.array(self.store)
        else:
            #with open(input_file) as raw_input_file:
                #csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                #for line in csv_file:
            for line in batcher_triples:
                e1 = line[0]
                r = line[1]
                e2 = line[2]
                if e1 in self.entity_vocab and e2 in self.entity_vocab:
                    e1 = self.entity_vocab[e1]
                    r = self.relation_vocab[r]
                    e2 = self.entity_vocab[e2]
                    self.store.append([e1,r,e2])
            self.store = np.array(self.store)
            fact_files = ['train.txt', 'test.txt', 'dev.txt', 'graph.txt']
            if os.path.isfile(self.input_dir+'/'+'full_graph.txt'):
                fact_files = ['full_graph.txt']
                print("Contains full graph")

            for f in fact_files:
            # for f in ['graph.txt']:
                with open(self.input_dir+'/'+f) as raw_input_file:
                    csv_file = csv.reader(raw_input_file, delimiter='\t')
                    for line in csv_file:
                        e1 = line[0]
                        r = line[1]
                        e2 = line[2]
                        if e1 in self.entity_vocab and e2 in self.entity_vocab:
                            e1 = self.entity_vocab[e1]
                            r = self.relation_vocab[r]
                            e2 = self.entity_vocab[e2]
                            self.store_all_correct[(e1, r)].add(e2)


    def yield_next_batch_train(self):
        '''
        randomly sample a batch of triples for training
        '''
        # while True:
        #     batch_idx = np.random.randint(0, self.store.shape[0], size=self.batch_size)
        #     batch = self.store[batch_idx, :]
        #     e1 = batch[:,0]
        #     r = batch[:, 1]
        #     e2 = batch[:, 2]
        #     all_e2s = []
        #     for i in range(e1.shape[0]):
        #         all_e2s.append(self.store_all_correct[(e1[i], r[i])])
        #     assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
        #     yield e1, r, e2, all_e2s

        np.random.shuffle(self.store)
        remaining_triples = self.store.shape[0]
        current_idx = 0
        while True:
            if remaining_triples == 0:
                np.random.shuffle(self.store)
            if remaining_triples - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                batch_idx = np.arange(current_idx, self.store.shape[0])
                remaining_triples = 0
            batch = self.store[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            yield e1, r, e2, all_e2s          


    def yield_next_batch_test(self):
        '''
        sample a batch of triples in order
        '''
        remaining_triples = self.store.shape[0]
        current_idx = 0
        while True:
            if remaining_triples == 0:
                return


            if remaining_triples - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                batch_idx = np.arange(current_idx, self.store.shape[0])
                remaining_triples = 0
            batch = self.store[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            yield e1, r, e2, all_e2s


class RelationEntityGrapher:
    '''
    for every entity, created an action tensor, which stores all available actions (out relation, entity)
    '''
    def __init__(self, triple_store, relation_vocab, entity_vocab, max_num_actions):

        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.triple_store = triple_store
        self.relation_vocab = relation_vocab
        self.entity_vocab = entity_vocab
        self.store = defaultdict(list)
        self.array_store = np.ones((len(entity_vocab), max_num_actions, 2), dtype=np.dtype('int32'))
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD
        self.masked_array_store = None

        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        self.create_graph()
        # print("KG constructed")

    def create_graph(self):
        '''
        action array "array_store" saves the action for each entity, each action is consists of the relation link and the next entity
        '''
        with open(self.triple_store) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                e1 = self.entity_vocab[line[0]]
                r = self.relation_vocab[line[1]]
                e2 = self.entity_vocab[line[2]]
                self.store[e1].append((r, e2)) # store record the out links for each entity

        for e1 in self.store:
            num_actions = 1
            self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP']
            self.array_store[e1, 0, 0] = e1
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]:
                    break
                self.array_store[e1,num_actions,0] = e2
                self.array_store[e1,num_actions,1] = r
                num_actions += 1
        del self.store
        self.store = None

    def return_next_actions(self, current_entities, start_entities, query_relations, answers, all_correct_answers, last_step, rollouts):

        ret = self.array_store[current_entities, :, :].copy() # (batch*num_rollouts, 250, 2)
        for i in range(current_entities.shape[0]):
            if current_entities[i] == start_entities[i]: # at the start point
                relations = ret[i, :, 1] # all linked relations
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i] , entities == answers[i]) # at the start point, mask out the direction link that can arrive at the answer
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD
            if last_step:
                # mask out other correct answers at the last step
                entities = ret[i, :, 0]
                relations = ret[i, :, 1]

                correct_e2 = answers[i]
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct_answers[int(i/rollouts)] and entities[j] != correct_e2:
                        entities[j] = self.ePAD
                        relations[j] = self.rPAD

        return ret




class Episode(object):

    def __init__(self, graph, data, params, extra_rollout=False):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher = params
        if extra_rollout:
            num_rollouts *= 5
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0
        start_entities, query_relation,  end_entities, all_answers = data
        self.no_examples = start_entities.shape[0]
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        start_entities = np.repeat(start_entities, self.num_rollouts) # shape (batch_size*num_rollouts,)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts) 
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers
        self.unbatched_query = query_relation


        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        self.past_entities = []

    def get_all_answers(self):
        return self.all_answers

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_initial_query(self):
        return self.unbatched_query

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward

    def get_acc_reward(self):
        acc_reward = []
        for entity in self.past_entities:
            #for entity in [self.current_entities]:
            reward = (entity == self.end_entities)

            # set the True and False values to the values of positive and negative rewards.
            condlist = [reward == True, reward == False]
            choicelist = [self.positive_reward, self.negative_reward]
            reward = np.select(condlist, choicelist)  # [B,]
            #if acc_reward is None:
            #    acc_reward = reward
            #else:
            #    acc_reward += reward
            acc_reward.append(reward)
            #print(reward.shape)
        return np.array(acc_reward)

    def __call__(self, action):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action]

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts )

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        self.past_entities.append(self.current_entities)
        return self.state


class env(object):
    def __init__(self, params, mode='train', batcher_triples=[], dev_triple=None):

        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        self.num_meta_tasks = params['num_meta_tasks']
        input_dir = params['data_input_dir']
        self.dev_batcher = []
        self.extra_rollout = params['extra_rollout']
        if mode == 'train':
            self.batcher = []
            if dev_triple is None:
                for _ in batcher_triples:
                    self.batcher.append(RelationEntityBatcher(input_dir=input_dir,
                                                              batch_size=params['batch_size'],
                                                              entity_vocab=params['entity_vocab'],
                                                              relation_vocab=params['relation_vocab'],
                                                              batcher_triples=_
                                                            ))
            else:
                for name, triples in batcher_triples.items():
                    if len(dev_triple) >0 and len(dev_triple[name])>0:
                        self.batcher.append(RelationEntityBatcher(input_dir=input_dir,
                                                                  batch_size=params['batch_size'],
                                                                  entity_vocab=params['entity_vocab'],
                                                                  relation_vocab=params['relation_vocab'],
                                                                  batcher_triples=triples
                                                                ))
                        self.dev_batcher.append(RelationEntityBatcher(input_dir=input_dir,
                                                                  batch_size=params['batch_size'],
                                                                  entity_vocab=params['entity_vocab'],
                                                                  relation_vocab=params['relation_vocab'],
                                                                  batcher_triples=triples
                                                                  #batcher_triples=dev_triple[name]
                                                                ))
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 mode =mode,
                                                 batch_size=min(params['batch_size']*10,128),
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 batcher_triples=batcher_triples
                                                 )

            self.total_no_examples = self.batcher.store.shape[0]
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'])

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            # yield_next_batch_train will return batched e1, r, e1 and all correct e2s
            batch_generaters = [task_batcher.yield_next_batch_train()
                                for task_batcher in self.batcher]
            dev_batch_generaters = [task_batcher.yield_next_batch_train()
                                for task_batcher in self.dev_batcher]
            batcher_data_size = [_.data_size for _ in self.batcher]
            batcher_pro = [_ / float(sum(batcher_data_size)) for _ in batcher_data_size]
            while True:
                ret_episodes = []
                #random.shuffle(batch_generaters)
                indexs = list(range(len(batch_generaters)))
                sel_indexs = np.random.choice(indexs, min(self.num_meta_tasks, len(batch_generaters)),
                                                replace=False, p=batcher_pro)
                for i in sel_indexs:
                    data = next(batch_generaters[i])
                    # print data[0].shape # (512,)
                    if len(dev_batch_generaters) == 0:
                        ret_episodes.append(Episode(self.grapher, data, params))
                    else:
                        dev_data = next(dev_batch_generaters[i])
                        #dev_data = next(batch_generaters[i])
                        if self.extra_rollout:
                            ret_episodes.append([Episode(self.grapher, data, params),Episode(self.grapher, dev_data, params, True)])
                        else:
                            ret_episodes.append([Episode(self.grapher, data, params),Episode(self.grapher, dev_data, params)])
                yield ret_episodes
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
