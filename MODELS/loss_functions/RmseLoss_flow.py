import torch
import json
import os


class RmseLoss_flow(torch.nn.Module):
    def __init__(self):
        super(RmseLoss_flow, self).__init__()

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["target"]
        obs_flow = y_obs[:, :, varTar_NN.index("00060_Mean")]
        sim_flow = y_sim.squeeze()
        if len(obs_flow[obs_flow == obs_flow]) > 0:
            mask_flow1 = obs_flow == obs_flow
            p = sim_flow[mask_flow1]
            t = obs_flow[mask_flow1]
            loss_flow_total = torch.sqrt(((p - t) ** 2).mean())  # RMSE item

        else:
            loss_flow_total = 0.0
        return loss_flow_total