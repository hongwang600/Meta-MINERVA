import torch
import numpy as np
torch.manual_seed(1)
np.random.seed(1)
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence
from collections import OrderedDict
from agent import Agent
from joblib import Parallel, delayed
import math
#from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
#                                       weighted_normalize)
#from maml_rl.utils.optimization import conjugate_gradient


def compute_new_params(agent, episodes, args):
    task_params = []
    for task_episode in episodes:
        task_params.append(agent.update_params(task_loss(agent, task_episode, args),
                                               args['alpha1']))
    return task_params

def compute_a_task_grad(agent, task_episode, args, i):
    cuda_id = i%2
    #cuda_id = 0
    #new_agent = Agent(args, cuda_id)
    #new_agent.load_state_dict(agent.state_dict())
    #new_agent.cuda(cuda_id)
    #print('before loss')
    #print(cuda_id, 'pass')
    this_task_loss=task_loss(agent, task_episode[0], args, cuda_id)
    new_params = agent.update_params(this_task_loss,
                                         args['alpha1'])
    #del agent
    #del this_task_loss
    #del new_agent
    new_agent = Agent(args, cuda_id)
    new_agent.cuda(cuda_id)
    new_agent.load_state_dict(new_params)
    new_loss = task_loss(new_agent, task_episode[1], args, cuda_id)
    this_task_grad = torch.autograd.grad(new_loss, new_agent.parameters())
    this_task_loss = new_loss.item()
    #del new_agent
    #print(task_grads[-1])
    del new_loss
    del new_agent
    del new_params
    torch.cuda.empty_cache()
    #print(this_task_grad, this_task_loss)
    #for grad in this_task_grad:
    #    grad.cuda(0)
    return this_task_grad, this_task_loss

def compute_tasks_grad(agent, episodes, args, i):
    results = []
    for this_episode in episodes:
        results.append(compute_a_task_grad(agent, this_episode, args, i))
    return results

def compute_grads(agent, episodes, args):
    task_grads = []
    task_losses = []
    chunk_size = 3
    num_chunks = math.ceil(len(episodes) / chunk_size)
    #results = Parallel(n_jobs=4, backend="threading")(delayed(compute_tasks_grad)(agent, episodes[chunk_id*chunk_size:(chunk_id+1)*chunk_size], args, chunk_id)
    #                             for chunk_id in range(num_chunks))
    results = [compute_a_task_grad(agent, _, args, 0) for _ in episodes]
    #for chunk_result in results:
    for chunk_result in [results]:
        for task_result in chunk_result:
            task_grads.append(task_result[0])
            task_losses.append(task_result[1])
    return task_grads, task_losses

def task_loss(agent, episode, args, cuda_id=0):
    query_rels = Variable(torch.from_numpy(episode.get_query_relation())).long().cuda(cuda_id)
    batch_size = query_rels.size()[0]
    state = episode.get_state()
    pre_rels = Variable(torch.ones(batch_size) * args['relation_vocab']['DUMMY_START_RELATION']).long().cuda(cuda_id)
    pre_states = agent.init_rnn_states(batch_size)

    record_action_probs = []
    record_probs = []
    for step in range(args['path_length']):
        next_rels = Variable(torch.from_numpy(state['next_relations'])).long().cuda(cuda_id)
        next_ents = Variable(torch.from_numpy(state['next_entities'])).long().cuda(cuda_id)
        curr_ents = Variable(torch.from_numpy(state['current_entities'])).long().cuda(cuda_id)

        probs, states = agent(next_rels, next_ents, pre_states, pre_rels, query_rels, curr_ents)
        record_probs.append(probs)
        action = torch.multinomial(probs, 1).detach()
        action_flat = action.data.squeeze()
        action_gather_indice = torch.arange(0, batch_size).long().cuda(cuda_id) * args['max_num_actions'] + action_flat
        action_prob = probs.view(-1)[action_gather_indice]
        record_action_probs.append(action_prob)
        chosen_relations = next_rels.view(-1)[action_gather_indice]

        pre_states = states
        pre_rels = chosen_relations
        state = episode(action_flat.cpu().numpy())

    rewards = episode.get_acc_reward()
    loss = agent.get_loss(rewards, record_action_probs, record_probs)
    #success_rate = np.sum(rewards) / batch_size
    #return batch_loss, avg_reward, success_rate
    return loss

def meta_step(agent, episodes, optim, args):
    """Meta-optimization step (ie. update of the initial parameters), based 
    on Trust Region Policy Optimization (TRPO, [4]).
    """
    #task_params = compute_new_params(agent, episodes, args)
    #grads = torch.autograd.grad(loss, agent.parameters)
    #grads = parameters_to_vector(grads)
    task_grads, task_losses = compute_grads(agent, episodes, args)
    '''
    losses = []
    for i, this_task_param in enumerate(task_params):
        new_agent = Agent(args)
        new_agent.cuda(cuda_id)
        new_agent.load_state_dict(this_task_param)
        losses.append(task_loss(new_agent, episodes[i], args))
    mean_loss = torch.mean(torch.stack(losses, dim=0))
    grads = torch.autograd.grad(mean_loss, agent.parameters(), allow_unused=True)
    '''
    optim.zero_grad()
    for grads in task_grads:
        #print('old param')
        for (name, param), grad in zip(agent.named_parameters(), grads):
            #print(param.data)
            #param.data -= args['alpha2']*grad
            if param.grad is not None:
                param.grad += grad.cuda(0)
            else:
                param.grad = grad.cuda(0)
        #print('new param')
        #for (name, param), grad in zip(agent.named_parameters(), grads):
            #print(param.data)
    nn.utils.clip_grad_norm(agent.parameters(), agent.grad_clip_norm)
    #for (name, param) in agent.named_parameters()::
    #    param.data -= args['alpha2']*param.grad
    #    del param.grad
    optim.step()
    for grads in task_grads:
        for grad in grads:
            del grad
    for param in agent.parameters():
        grad = param.grad
        param.grad = None
        del grad
    torch.cuda.empty_cache()
    agent.update_steps += 1
    return np.mean(task_losses)
