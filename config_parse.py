import argparse
import yaml
import sys
from utils.log import logger
from easydict import EasyDict as edict

def parse_arguments(args):
    """
    Parse arguments from command line.
    Syntax: --key1.key2.key3=value --> value
            --key1.key2.key3=      --> None
            --key1.key2.key3       --> True
            --key1.key2.key3!      --> False
    Input: sys.argv[1:]
    """
    opt_cmd = {}
    for arg in args:
        assert(arg.startswith("--"))
        if "=" not in arg[2:]:
            key_str,value = (arg[2:-1],"false") if arg[-1]=="!" else (arg[2:],"true")
        else:
            key_str,value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub: opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub,keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    opt_cmd = edict(opt_cmd)
    return opt_cmd

# opt_parent: parent option
# opt: child option, opt may have extra and different keys/values to opt_parent
def override_edict(opt_parent, opt, safe_check=False):
    for key, value in opt.items():
        if isinstance(value, dict):
            # parse child options (until leaf nodes are reached)
            opt_parent[key] = override_edict(opt_parent.get(key,dict()), value)
        else:
            # ensure command line argument to override is also in yaml file
            if key not in opt_parent:
                logger.info("add {}: {}".format(key, value))
            elif key in opt_parent:
                logger.info("override {}: {} -> {}".format(key, opt_parent[key], value))
            opt_parent[key] = value
    return opt_parent

def parse_yaml_file(file_path):
    logger.title("parse yaml: {}".format(file_path))
    with open(file_path) as file:
        data = edict(yaml.safe_load(file))
    if data.get('_parent_'):
        parent_path = data.get('_parent_')
        parent_data = parse_yaml_file(parent_path)  # load base yaml
        data = override_edict(parent_data, data)
    return data

def config():
    args = parse_arguments(sys.argv[1:])
    data = parse_yaml_file(args.config)
    data = override_edict(data, args)
    return data

### TODO: save option file

if __name__ == '__main__':
    cfg = config()
    print(cfg)
