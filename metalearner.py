import torch
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence
from collections import OrderedDict
from agent import Agent

#from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
#                                       weighted_normalize)
#from maml_rl.utils.optimization import conjugate_gradient


def compute_new_params(agent, episodes, args):
    task_params = []
    for task_episode in episodes:
        task_params.append(agent.update_params(task_loss(agent, task_episode, args),
                                               args['alpha1']))
    return task_params

def compute_loss(agent, episodes, args):
    losses = []
    for task_episode in epidoses:
        losses.append(task_loss(agent, task_episode, args))
    return torch.mean(torch.stack(losses, dim=0))

def task_loss(agent, episode, args):
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
    loss = agent.get_loss(rewards, record_action_probs, record_probs)
    #success_rate = np.sum(rewards) / batch_size
    #return batch_loss, avg_reward, success_rate
    return loss

def meta_step(agent, episodes, args):
    """Meta-optimization step (ie. update of the initial parameters), based 
    on Trust Region Policy Optimization (TRPO, [4]).
    """
    task_params = compute_new_params(agent, episodes, args)
    #grads = torch.autograd.grad(loss, agent.parameters)
    #grads = parameters_to_vector(grads)
    losses = []
    for i, this_task_param in enumerate(task_params):
        new_agent = Agent(args)
        new_agent.cuda()
        new_agent.load_state_dict(this_task_param)
        losses.append(task_loss(new_agent, episodes[i], args))
    mean_loss = torch.mean(torch.stack(losses, dim=0))
    grads = torch.autograd.grad(mean_loss, agent.parameters(), allow_unused=True)
    for (name, param), grad in zip(agent.named_parameters(), grads):
        print(grad)
        param = param - args['alpha2'] * grad
