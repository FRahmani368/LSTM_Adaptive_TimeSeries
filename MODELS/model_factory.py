import torch.nn

from MODELS.NN_models.LSTM_models import CudnnLstmModel
from MODELS.NN_models.MLP_models import MLPmul

from core.utils.small_codes import source_flow_calculation

# import MODELS
class create_NN_models(torch.nn.Module):
    def __init__(self, args):
        super(create_NN_models, self).__init__()
        self.args = args
        self.get_model()

    def get_model(self) -> None:
        self.nx = len(self.args["varT_NN"] + self.args["varC_NN"])

        # output size of NN
        self.ny = len(self.args["target"])  # + self.ny_PET

        # NN_model_initialization
        if self.args["NN_model_name"] == "LSTM":
            self.NN_model = CudnnLstmModel(nx=self.nx,
                                           ny=self.ny,
                                           hiddenSize=self.args["hidden_size"],
                                           args=self.args,
                                           dr=self.args["dropout"],
                                           )
        elif self.args["NN_model_name"] == "MLP":
            self.NN_model = MLPmul(self.args, nx=self.nx, ny=self.ny)
        else:
            print("NN model type has not been defined")
            exit()

    def breakdown_params(self, params_all):
        params_dict = dict()
        params_hydro_model = params_all[:, :, :self.ny_hydro]
        params_temp_model = params_all[:, :, self.ny_hydro: (self.ny_hydro + self.ny_temp)]
        # if self.ny_PET > 0:
        #     params_dict["params_PET_model"] = torch.sigmoid(params_all[-1, :, (self.ny_hydro + self.ny_temp):])
        # else:
        #     params_dict["params_PET_model"] = None


        # Todo: I should separate PET model output from hydro_model and temp_model.
        #  For now, evap is calculated in both models individually (with same method)

        if self.args['hydro_model_name'] != "None":
            # hydro params
            params_dict["hydro_params_raw"] = torch.sigmoid(
                params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.args["nmul"]]).view(
                params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
                self.args["nmul"])
            # routing params
            if self.args["routing_hydro_model"] == True:
                params_dict["conv_params_hydro"] = torch.sigmoid(
                    params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.args["nmul"]:])
            else:
                params_dict["conv_params_hydro"] = None

        if self.args['temp_model_name'] != "None":
            # if lat_temp_adj is True --> dim:[days, basins, 5,  nmul]   , because there are 5 params in temp model
            if self.args["lat_temp_adj"] == True:
                params_dict["temp_params_raw"] = torch.sigmoid(
                    params_temp_model[:, :,
                    :(len(self.temp_model.parameters_bound) + len(self.temp_model.lat_adj_params_bound)) * self.args[
                        "nmul"]]).view(
                    params_temp_model.shape[0], params_temp_model.shape[1],
                    len(self.temp_model.parameters_bound) + len(self.temp_model.lat_adj_params_bound),
                    self.args["nmul"])
            # if lat_temp_adj is False --> dim:[days, basins, 5 + 1,  nmul]
            else:
                params_dict["temp_params_raw"] = torch.sigmoid(
                    params_temp_model[:, :, :len(self.temp_model.parameters_bound) * self.args["nmul"]]).view(
                    params_temp_model.shape[0], params_temp_model.shape[1], len(self.temp_model.parameters_bound),
                    self.args["nmul"])
            # convolution parameters for ss and gw temp calculation
            if self.args["routing_temp_model"] == True:
                params_dict["conv_params_temp"] = torch.sigmoid(params_temp_model[-1, :, -len(self.temp_model.conv_temp_model_bound):])
            else:
                print("it has not been defined yet what approach should be taken in place of conv")
                exit()
        return params_dict


    def forward(self, dataset_dictionary_sample):
        y_sim = self.NN_model(dataset_dictionary_sample["inputs_NN_scaled"])   # [self.args["warm_up"]:, :, :]

        return y_sim  # combining both dictionaries
