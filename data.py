import json

def read_json_data(file_name):
    with open(file_name) as f_in:
        return json.load(f_in)

def construct_data(args):
    input_dir = args['data_input_dir']
    meta_train_data = read_json_data(input_dir+'/'+args['meta_train_file'])
    meta_dev_data = read_json_data(input_dir+'/'+args['meta_dev_file'])
    meta_test_data = read_json_data(input_dir+'/'+args['meta_test_file'])
    few_shot_data = read_json_data(input_dir+'/'+args['few_shot_file'])
    return [meta_train_data, meta_dev_data, meta_test_data, few_shot_data]

def concat_data(data):
    ret_data = []
    for k in data:
        ret_data += data[k]
    return ret_data
