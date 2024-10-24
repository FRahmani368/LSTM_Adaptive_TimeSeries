"""code to read the config file"""
import os

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    print("YAML Module not found.")

"""Local terminal path"""
"""pycharm path"""
config_path_NN_model = "config/config_NN_model.yaml"
yaml = YAML(typ="safe")
path_NN_model = os.path.realpath(config_path_NN_model)
stream_NN_model = open(path_NN_model, "r")
config_NN_model = yaml.load(stream_NN_model)

