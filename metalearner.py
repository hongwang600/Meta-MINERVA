import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

#from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
#                                       weighted_normalize)
#from maml_rl.utils.optimization import conjugate_gradient


def compute_new_params(agent, episodes, args):
    task_params = []
    for task_episode in epidoses:
        task_params.append(agent.update_params(task_loss(agent, task_episode),
                                               args['alpha1']))
    return task_params

def compute_loss(agent, episodes):
    losses = []
    for task_episode in epidoses:
        losses.append(task_loss(agent, task_episode))
    return torch.mean(torch.stack(losses, dim=0))

def task_loss(agent, episode):
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
    grads = agent.get_grads(rewards, record_action_probs, record_probs)
    #success_rate = np.sum(rewards) / batch_size
    #return batch_loss, avg_reward, success_rate
    return grads

def step(agent, episodes, args):
    """Meta-optimization step (ie. update of the initial parameters), based 
    on Trust Region Policy Optimization (TRPO, [4]).
    """
    task_params = compute_new_params(agent, episodes, args)
    grads = torch.autograd.grad(loss, agent.parameters)
    grads = parameters_to_vector(grads)

    # Compute the step direction with Conjugate Gradient
    hessian_vector_product = self.hessian_vector_product(episodes,
        damping=cg_damping)
    stepdir = conjugate_gradient(hessian_vector_product, grads,
        cg_iters=cg_iters)

    # Compute the Lagrange multiplier
    shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
    lagrange_multiplier = torch.sqrt(shs / max_kl)

    step = stepdir / lagrange_multiplier

    # Save the old parameters
    old_params = parameters_to_vector(self.policy.parameters())

    # Line search
    step_size = 1.0
    for _ in range(ls_max_steps):
        vector_to_parameters(old_params - step_size * step,
                             self.policy.parameters())
        loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
        improve = loss - old_loss
        if (improve.item() < 0.0) and (kl.item() < max_kl):
            break
        step_size *= ls_backtrack_ratio
    else:
        vector_to_parameters(old_params, self.policy.parameters())
