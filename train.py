from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import logging
from tqdm import tqdm
import os
from collections import defaultdict
import shutil
import random
import copy
from collections import OrderedDict

from tensorboardX import SummaryWriter

from env import RelationEntityBatcher, RelationEntityGrapher, env
from options import read_options
from agent import Agent
from data import construct_data, concat_data, get_id_relation, build_vocab
from metalearner import meta_step
 
# read parameters
random.seed(1)
args = read_options()
# logging
logger = logging.getLogger(args['id'])
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logfile = logging.FileHandler(args['log_path'], 'w')
logfile.setFormatter(formatter)
logger.addHandler(logfile)

def train_one_episode(agent, episode):
    query_rels = Variable(torch.from_numpy(episode.get_query_relation())).long().cuda()
    batch_size = query_rels.size()[0]
    state = episode.get_state()
    pre_rels = Variable(torch.ones(batch_size) * args['relation_vocab']['DUMMY_START_RELATION']).long().cuda()
    pre_states = agent.init_rnn_states(batch_size)

    record_action_probs = []
    record_probs = []
    for step in range(args['path_length']):
        next_rels = Variable(torch.from_numpy(state['next_relations'])).long().cuda()
        next_ents = Variable(torch.from_numpy(state['next_entities'])).long().cuda()
        curr_ents = Variable(torch.from_numpy(state['current_entities'])).long().cuda()

        probs, states = agent(next_rels, next_ents, pre_states, pre_rels, query_rels, curr_ents)
        record_probs.append(probs)
        action = torch.multinomial(probs, 1).detach()
        action_flat = action.data.squeeze()
        action_gather_indice = torch.arange(0, batch_size).long().cuda() * args['max_num_actions'] + action_flat
        action_prob = probs.view(-1)[action_gather_indice]
        record_action_probs.append(action_prob)
        chosen_relations = next_rels.view(-1)[action_gather_indice]

        pre_states = states
        pre_rels = chosen_relations
    state = episode(action_flat.cpu().numpy())

    rewards = episode.get_reward()
    batch_loss, avg_reward = agent.update(rewards, record_action_probs, record_probs)
    success_rate = np.sum(rewards) / batch_size
    return batch_loss, avg_reward, success_rate

def train(args):
    data = construct_data(args)
    id_rels = get_id_relation(args)
    my_vocab = build_vocab(args)
    #print(len(id_rels))
    train_data, dev_data, meta_dev_data, few_shot_dev_data = data
    #concated_train_data = concat_data(train_data)
    #concated_dev_data = concat_data(dev_data)
    #random.shuffle(concated_train_data)
    #random.shuffle(concated_dev_data)
    #print(concated_dev_data)
    logger.info('start training')

    # build the train and validation environment here
    #train_env = env(args, mode='train', batcher_triples=concated_train_data)
    #dev_env = env(args, mode='dev', batcher_triples=concated_dev_data)

    #print(len(train_data.values()))
    train_env = env(args, mode='train', batcher_triples=train_data.values())
    #print('load all envs')

    if os.path.exists('logs/' + args['id']):
        shutil.rmtree('logs/' + args['id'], ignore_errors=True)

    writer = SummaryWriter('logs/' + args['id'])

    beta = args['beta']

    # build the agent here
    agent = Agent(args)
    agent.cuda()
    #print(OrderedDict(agent.named_parameters()))

    #meta_learner = MetaLearner(train_env, agent)

    for episode in train_env.get_episodes():
        #print('get episode')
        meta_step(agent, episode, args)
        batch_loss, avg_reward, success_rate = train_one_episode(agent, episode)
        writer.add_scalar('batch_loss', batch_loss, agent.update_steps)
        writer.add_scalar('avg_reward', avg_reward, agent.update_steps)

        logger.info('batch_loss at iter %d: %f' % (agent.update_steps, batch_loss))
        logger.info('success_rate at iter %d: %f' % (agent.update_steps, success_rate))

        if agent.update_steps % args['eval_every'] == 0:
            test(agent, args, writer, dev_env)

        if agent.update_steps > args['total_iterations']:
            agent.save(args['save_path'])
            break
    meta_test(agent, args, writer, few_shot_dev_data, meta_dev_data)

def single_task_meta_test(ori_agent, args, few_shot_data, test_data, training_step):
    agent = Agent(args)
    agent.cuda()
    agent.load_state_dict(ori_agent.state_dict())
    #print(agent.update_steps)
    agent.update_steps = 0
    #print(len(few_shot_data), len(test_data))
    train_env = env(args, mode='train', batcher_triples=few_shot_data)
    test_env = env(args, mode='dev', batcher_triples=test_data)
    test_scores = []
    test_scores.append(test(agent, args, None, test_env))
    for episode in train_env.get_episodes():
        batch_loss, avg_reward, success_rate = train_one_episode(agent, episode)
        #if agent.update_steps % args['eval_every'] == 0:
        test_scores.append(test(agent, args, None, test_env))
        if agent.update_steps == training_step:
            break
    return np.array(test_scores)

def meta_test(agent, args, writer, few_shot_data, test_data):
    num_meta_step = args['meta_step']
    task_results = np.zeros([num_meta_step+1, 6])
    for task in few_shot_data:
        task_results += single_task_meta_test(agent, args, few_shot_data[task],
                                              test_data[task], num_meta_step)
    task_results /= len(few_shot_data)
    pre_str = 'meta_'
    for i in range(len(task_results)):
        writer.add_scalar(pre_str+'Hits1', task_results[i][0], i)
        writer.add_scalar(pre_str+'Hits3', task_results[i][1], i)
        writer.add_scalar(pre_str+'Hits5', task_results[i][2], i)
        writer.add_scalar(pre_str+'Hits10', task_results[i][3], i)
        writer.add_scalar(pre_str+'Hits20', task_results[i][4], i)
        writer.add_scalar(pre_str+'AUC', task_results[i][5], i)

def test(agent, args, writer, test_env, mode='dev', print_paths=False, is_meta_test=False):

    # index to relation/entity names
    id_rels = {}
    # id_ents = {}
    for key, value in args['relation_vocab'].items():
        id_rels[value] = key
    # for key, value in args['entity_vocab'].items():
    #     id_ents[value] = key

    agent.eval()
    answers = []
    feed_dict = {}
    all_final_reward_1 = 0
    all_final_reward_3 = 0
    all_final_reward_5 = 0
    all_final_reward_10 = 0
    all_final_reward_20 = 0
    auc = 0

    ranks_per_rel = defaultdict(list)
    aps_per_rel = defaultdict(list)

    total_examples = test_env.total_no_examples
    for episode in tqdm(test_env.get_episodes()):
        batch_size = episode.no_examples
        query_rels = Variable(torch.from_numpy(episode.get_query_relation())).long().cuda()
        beam_probs = np.zeros((batch_size * args['test_rollouts'], 1))
        state = episode.get_state()
        pre_rels = Variable(torch.ones(batch_size * args['test_rollouts']) * args['relation_vocab']['DUMMY_START_RELATION']).long().cuda()
        pre_states = agent.init_rnn_states(batch_size * args['test_rollouts'])

        for step in range(args['path_length']):
            next_rels = Variable(torch.from_numpy(state['next_relations'])).long().cuda()
            next_ents = Variable(torch.from_numpy(state['next_entities'])).long().cuda()
            curr_ents = Variable(torch.from_numpy(state['current_entities'])).long().cuda()

            scores, states = agent(next_rels, next_ents, pre_states, pre_rels, query_rels, curr_ents)
            scores = scores.detach()
            log_scores = torch.log(scores + 1e-8)

            # beam search
            k = args['test_rollouts']
            new_scores = log_scores.data.cpu().numpy() + beam_probs # use global scores for beam search

            if step == 0:
                # for initial beam, items in the same beam are equavalent
                idx = np.argsort(new_scores)
                idx = idx[:, -k:]
                ranged_idx = np.tile([b for b in range(k)], batch_size)
                idx = idx[np.arange(k*batch_size), ranged_idx]
            else:
                _ = new_scores.reshape(-1, k * args['max_num_actions'])
                idx = np.argsort(_, axis=1)
                idx = idx[:,-k:]
                idx = idx.reshape((-1))

            # print idx.shape # (b*test_rollouts,)
            y = idx // args['max_num_actions'] # idx in each beam
            x = idx % args['max_num_actions'] # action_idx
            y += np.repeat([b*k for b in range(batch_size)], k) # convert beam idx to global idx
            state['current_entities'] = state['current_entities'][y]
            state['next_relations'] = state['next_relations'][y,:]
            state['next_entities'] = state['next_entities'][y,:]
            new_states = []
            for i in range(len(states)):
                new_states.append((states[i][0][y,:], states[i][1][y,:]))

            test_action_idx = x
            chosen_relations = state['next_relations'][np.arange(batch_size*k), x]
            beam_probs = new_scores[y, x]
            beam_probs = beam_probs.reshape((-1, 1))

            pre_rels = Variable(torch.from_numpy(chosen_relations)).long().cuda()
            pre_states = new_states
            state = episode(test_action_idx)
        log_probs = beam_probs

        rewards = episode.get_reward()
        if writer is not None:
            writer.add_scalar('Dev Avg Reward', np.mean(rewards), agent.update_steps)

        reward_reshape = np.reshape(rewards, (batch_size, args['test_rollouts']))
        log_probs = np.reshape(log_probs, (batch_size, args['test_rollouts']))
        sorted_indx = np.argsort(-log_probs)
        final_reward_1 = 0
        final_reward_3 = 0
        final_reward_5 = 0
        final_reward_10 = 0
        final_reward_20 = 0
        AP = 0
        ce = episode.state['current_entities'].reshape((batch_size, args['test_rollouts']))
        se = episode.start_entities.reshape((batch_size, args['test_rollouts']))

        # calculate scores for each query relations
        relation_ids = episode.get_initial_query()

        for b in range(batch_size):

            rel_name = id_rels[relation_ids[b]]

            answer_pos = None
            seen = set()
            pos=0
            if args['pool'] == 'max':
                for r in sorted_indx[b]:
                    if reward_reshape[b,r] == args['positive_reward']:
                        answer_pos = pos
                        break
                    if ce[b, r] not in seen:
                        seen.add(ce[b, r])
                        pos += 1
            else:
                raise NotImplementedError('Not implemented yet')

            # add the ranks to ranks_per_rel
            if answer_pos == None:
                ranks_per_rel[rel_name].append(9999)
                aps_per_rel[rel_name].append(0)
            else:
                ranks_per_rel[rel_name].append(answer_pos + 1)
                aps_per_rel[rel_name].append(1.0/((answer_pos+1)))

            if answer_pos != None:
                if answer_pos < 20:
                    final_reward_20 += 1
                    if answer_pos < 10:
                        final_reward_10 += 1
                        if answer_pos < 5:
                            final_reward_5 += 1
                            if answer_pos < 3:
                                final_reward_3 += 1
                                if answer_pos < 1:
                                    final_reward_1 += 1
            if answer_pos == None:
                AP += 0
            else:
                AP += 1.0/((answer_pos+1))

        all_final_reward_1 += final_reward_1
        all_final_reward_3 += final_reward_3
        all_final_reward_5 += final_reward_5
        all_final_reward_10 += final_reward_10
        all_final_reward_20 += final_reward_20
        auc += AP

    all_final_reward_1 /= total_examples
    all_final_reward_3 /= total_examples
    all_final_reward_5 /= total_examples
    all_final_reward_10 /= total_examples
    all_final_reward_20 /= total_examples
    auc /= total_examples

    if mode == 'test':
        hits10_per_rel = {}
        hits5_per_rel = {}
        hits1_per_rel = {}
        mrr_per_rel = {}
        for key, ranks_ in ranks_per_rel.items():
            mrr_per_rel[key] = np.mean(1./np.array(ranks_))
            hits10 = []
            hits5 = []
            hits1 = []
            for rank in ranks_:
                if rank <= 10:
                    hits10.append(1.0)
                else:
                    hits10.append(0.0)

                if rank <= 5:
                    hits5.append(1.0)
                else:
                    hits5.append(0.0)

                if rank <= 1:
                    hits1.append(1.0)
                else:
                    hits1.append(0.0)

            hits10_per_rel[key] = np.mean(hits10)
            hits5_per_rel[key] = np.mean(hits5)
            hits1_per_rel[key] = np.mean(hits1)

        json.dump(hits10_per_rel, open('results/' + args['id'] + '_hits10_per_rel', 'w'))
        json.dump(hits5_per_rel, open('results/' + args['id'] + '_hits5_per_rel', 'w'))
        json.dump(hits1_per_rel, open('results/' + args['id'] + '_hits1_per_rel', 'w'))
        json.dump(mrr_per_rel, open('results/' + args['id'] +'_mrr_per_rel', 'w'))
        logger.info('Per relation results saved')

    # if save_model:
    #     if all_final_reward_10 > self.max_hits_at_10:
    #         self.max_hits_at_10 = all_final_reward_10
    #         self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

    pre_str=''
    if is_meta_test:
        pre_str = 'meta_'
    logger.info(pre_str+"Hits@1: {0:7.4f}".format(all_final_reward_1))
    logger.info(pre_str+"Hits@3: {0:7.4f}".format(all_final_reward_3))
    logger.info(pre_str+"Hits@5: {0:7.4f}".format(all_final_reward_5))
    logger.info(pre_str+"Hits@10: {0:7.4f}".format(all_final_reward_10))
    logger.info(pre_str+"Hits@20: {0:7.4f}".format(all_final_reward_20))
    logger.info(pre_str+"auc: {0:7.4f}".format(auc))

    if writer is not None:
        writer.add_scalar(pre_str+'Hits10', all_final_reward_10, agent.update_steps)
        writer.add_scalar(pre_str+'Hits1', all_final_reward_1, agent.update_steps)
        writer.add_scalar(pre_str+'Hits3', all_final_reward_3, agent.update_steps)
        writer.add_scalar(pre_str+'Hits5', all_final_reward_5, agent.update_steps)
        writer.add_scalar(pre_str+'Hits20', all_final_reward_20, agent.update_steps)
        writer.add_scalar(pre_str+'AUC', auc, agent.update_steps)

    agent.train()
    return [all_final_reward_1,all_final_reward_3,all_final_reward_5,
            all_final_reward_10,all_final_reward_20,auc]

if __name__ == '__main__':
    if args['test']:
        agent = Agent(args)
        agent.cuda()
        agent.load(args['save_path'])
        test_env = env(args, mode='test')
        test(agent, args, writer, test_env, mode='test')
    else:
        train(args)


