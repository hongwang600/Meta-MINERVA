import json
import random

random.seed(1)
train_ratio = 0.8
few_shot_size = 40

relation_vocab_file = 'vocab/relation_vocab.json'
entity_vocab_file = 'vocab/entity_vocab.json'
train_file = 'train.txt'
dev_file = 'dev.txt'
test_file = 'test.txt'

def load_data(file_name):
    data = []
    with open(file_name) as file_in:
        for line in file_in:
            line = line.strip().split('\t')
            data.append(line)
    return data

def get_all_relations(data):
    all_relations = []
    for sample in data:
        if sample[1] not in all_relations:
            all_relations.append(sample[1])
    return all_relations

def save_data(data, file_name):
    with open(file_name, 'w') as file_in:
        json.dump(data, file_in)

def check_pairs(data_1, data_2):
    to_pop = []
    for k in data_1:
        if len(data_1[k])==0 or len(data_2[k]) == 0:
           to_pop.append(k)
    for k in to_pop:
        data_1.pop(k)
        data_2.pop(k)

if __name__ == '__main__':
    with open(relation_vocab_file) as file_in:
        relation_vocab = json.load(file_in)
    with open(entity_vocab_file) as file_in:
        entity_vocab = json.load(file_in)
    train_data = load_data(train_file)
    dev_data = load_data(dev_file)
    test_data = load_data(test_file)
    all_relations = get_all_relations(train_data)
    #print(relation_vocab[train_data[0][1]])
    #print(len(all_relations))
    train_relation_size = int(len(all_relations) * train_ratio)
    train_relations = random.sample(all_relations, train_relation_size)
    rest_relations = [relation for relation in all_relations if relation not in train_relations]
    dev_relations = random.sample(rest_relations, int(len(rest_relations)/2))
    test_relations = [relation for relation in rest_relations if relation not in dev_relations]
    pre_train_data = {}
    pre_dev_data = {}
    few_shot_dev_data = {}
    few_shot_test_data = {}
    meta_test_data = {}
    meta_dev_data = {}
    for relation in train_relations:
        pre_train_data[relation] = []
        pre_dev_data[relation] = []
    for relation in dev_relations:
        few_shot_dev_data[relation] = []
        meta_dev_data[relation] = []
    for relation in test_relations:
        few_shot_test_data[relation] = []
        meta_test_data[relation] = []
    for sample in train_data:
        if sample[1] in train_relations:
            pre_train_data[sample[1]].append(sample)
        elif sample[1] in dev_relations:
            few_shot_dev_data[sample[1]].append(sample)
        else:
            few_shot_test_data[sample[1]].append(sample)
    for sample in dev_data:
        if sample[1] in train_relations:
            pre_dev_data[sample[1]].append(sample)
        elif sample[1] in dev_relations:
            meta_dev_data[sample[1]].append(sample)
    for sample in test_data:
        if sample[1] in test_relations:
            meta_test_data[sample[1]].append(sample)
    check_pairs(pre_train_data, pre_dev_data)
    check_pairs(few_shot_dev_data, meta_dev_data)
    check_pairs(few_shot_test_data, meta_test_data)
    print(len(pre_train_data), len(meta_dev_data), len(meta_test_data))
    save_data(pre_train_data, 'train.json')
    save_data(pre_dev_data, 'dev.json')
    save_data(meta_dev_data, 'meta_dev.json')
    save_data(meta_test_data, 'meta_test.json')
    save_data(few_shot_dev_data, 'few_shot_dev.json')
    save_data(few_shot_test_data, 'few_shot_test.json')
