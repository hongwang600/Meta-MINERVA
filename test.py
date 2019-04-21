from data import construct_data, concat_data, get_id_relation, build_vocab
from train import meta_test
from agent import Agent
from options import read_options
from tensorboardX import SummaryWriter

args = read_options()

if __name__ == '__main__':
    agent = Agent(args)
    agent.cuda()
    agent.load(args['save_path'][:-9])
    data = construct_data(args)
    train_data, dev_data, meta_dev_data, few_shot_dev_data = data
    writer = SummaryWriter(args['log_dir'] + args['id'])
    meta_test(agent, args, writer, few_shot_dev_data, meta_dev_data)
