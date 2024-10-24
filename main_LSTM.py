import sys
import os
import torch
from config.read_configurations import config_NN_model as config
from core.utils.randomseed_config import randomseed_config
from core.utils.small_codes import create_output_dirs
from MODELS.model_factory import create_NN_models
from MODELS import train_test
sys.path.append('../')

def main_LSTM(args):
    randomseed_config(seed=args["randomseed"][0])
    # Creating output directories and adding it to args
    args = create_output_dirs(args)

    if "Train" in args["Action"]:  # training mode
        model = create_NN_models(args)
        optim = torch.optim.Adadelta(model.parameters())
        train_test.train_NN_model(
            args=args,
            model=model,
            optim=optim
        )
    if "Test" in args["Action"]:  # testing mode
        modelFile = os.path.join(args["out_dir"], "model_Ep" + str(args["EPOCH_testing"]) + ".pt")
        model = torch.load(modelFile)
        train_test.test_NN_model(
            args=args,
            model=model
        )


if __name__ == "__main__":
    args = config
    main_LSTM(args)
    print("END")