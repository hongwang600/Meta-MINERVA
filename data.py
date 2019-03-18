import json

def read_json_data(file_name):
    with open(file_name) as f_in:
        return json.load(f_in)

def construct_data(args):
    input_dir = args['data_input_dir']
    train_data = read_json_data(input_dir+'/'+args['train_file'])
    dev_data = read_json_data(input_dir+'/'+args['dev_file'])
    meta_dev_data = read_json_data(input_dir+'/'+args['meta_dev_file'])
    few_shot_dev_data = read_json_data(input_dir+'/'+args['few_shot_dev_file'])
    return [train_data, dev_data, meta_dev_data, few_shot_dev_data]

def concat_data(data):
    ret_data = []
    for k in data:
        ret_data += data[k]
    return ret_data
