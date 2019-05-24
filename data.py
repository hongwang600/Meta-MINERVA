import json
import gluonnlp

def read_json_data(file_name):
    with open(file_name) as f_in:
        return json.load(f_in)

def construct_data(args):
    input_dir = args['data_input_dir']
    train_data = read_json_data(input_dir+'/'+args['train_file'])
    dev_data = read_json_data(input_dir+'/'+args['dev_file'])
    meta_dev_data = read_json_data(input_dir+'/'+args['meta_dev_file'])
    few_shot_dev_data = read_json_data(input_dir+'/'+args['few_shot_dev_file'])
    #for task in few_shot_dev_data:
    #    few_shot_dev_data[task] = few_shot_dev_data[task][:args['few_shot_size']]
    few_shot_dev_data = few_shot_dev_data
    train_data.update(few_shot_dev_data)
    return [train_data, dev_data, meta_dev_data, few_shot_dev_data]

def concat_data(data):
    ret_data = []
    for k in data:
        ret_data += data[k]
    return ret_data

def get_id_relation(args):
    id_rels = {}
    # id_ents = {}
    for key, value in args['relation_vocab'].items():
        id_rels[value] = key
    return id_rels

def tokenize_relation(rel):
    tokens = []
    relations = rel.split('.')
    for this_relation in relations:
        for _ in this_relation.split('/')[-3:]:
            tokens+=(_.split('_'))
        tokens.append('[SEP]')
    return tokens[:-1]

def build_vocab(args):
    tokens = []
    #print(len(args['relation_vocab']))
    for key, value in args['relation_vocab'].items():
        tokens += tokenize_relation(key)
    counter = gluonnlp.data.count_tokens(tokens)
    my_vocab = gluonnlp.Vocab(counter)
    glove = gluonnlp.embedding.create('glove', source='glove.6B.100d')
    my_vocab.set_embedding(glove)
    return my_vocab
