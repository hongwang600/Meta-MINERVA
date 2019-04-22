from __future__ import absolute_import
from __future__ import division
import argparse
import json

def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input_dir", default="FB15k-237", type=str)
    #parser.add_argument("--input_file", default="meta_train.txt", type=str)
    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--meta_dev_file", default="meta_dev.json", type=str)
    parser.add_argument("--meta_step", default=50, type=int)
    parser.add_argument("--few_shot_dev_file", default="few_shot_dev.json", type=str)
    parser.add_argument("--few_shot_size", default=1, type=int)
    parser.add_argument("--num_meta_tasks", default=10, type=int)
    parser.add_argument("--create_vocab", default=0, type=int)
    parser.add_argument("--vocab_dir", default="", type=str)
    parser.add_argument("--max_num_actions", default=200, type=int)
    parser.add_argument("--path_length", default=3, type=int)
    parser.add_argument("--hidden_size", default=50, type=int)
    parser.add_argument("--embed_size", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--grad_clip_norm", default=5, type=int)
    parser.add_argument("--l2_reg_const", default=1e-2, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--alpha1", default=1e-1, type=float)
    parser.add_argument("--alpha2", default=1e-3, type=float)
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument("--positive_reward", default=1.0, type=float)
    parser.add_argument("--negative_reward", default=0, type=float)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--log_dir", default="./ablation_logs/", type=str)
    # parser.add_argument("--log_file_name", default="reward.txt", type=str)
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--num_rollouts", default=20, type=int)
    parser.add_argument("--test_rollouts", default=100, type=int)
    parser.add_argument("--policy_layers", default=1, type=int)
    parser.add_argument("--model_dir", default='', type=str)
    parser.add_argument("--base_output_dir", default='', type=str)
    parser.add_argument("--total_iterations", default=2000, type=int)

    parser.add_argument("--Lambda", default=0.0, type=float)
    parser.add_argument("--pool", default="max", type=str)
    parser.add_argument("--eval_every", default=50, type=int)
    # parser.add_argument("--use_entity_embeddings", aciton='store_true')
    parser.add_argument("--train_entity_embeddings", action='store_true')
    parser.add_argument("--train_relation_embeddings", action='store_true')
    parser.add_argument("--model_load_dir", default="", type=str)
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--extra_rollout", action='store_true')
    parser.add_argument("--new_reward", action='store_true')

    parser.add_argument("--id", default='vanilla', type=str)

    arg_dic = vars(parser.parse_args())
    arg_dic['log_path'] = arg_dic['log_dir'] + arg_dic['id'] + '.txt'

    arg_dic['save_path'] = 'models/' + arg_dic['id']

    # read KB vocab
    arg_dic['vocab_dir'] = arg_dic['data_input_dir'] + '/vocab'
    arg_dic['relation_vocab'] = json.load(open(arg_dic['vocab_dir'] + '/relation_vocab.json'))
    arg_dic['entity_vocab'] = json.load(open(arg_dic['vocab_dir'] + '/entity_vocab.json'))

    return arg_dic

