import os
import numpy as np
import pandas as pd
import torch
import math
import time
from post import stat, plot
import matplotlib.pyplot as plt
from core.load_data.normalizing import init_norm_stats
from core.load_data.dataFrame_loading import loadData
from core.load_data.data_prep import (
    No_iter_nt_ngrid,
    take_sample_train,
    take_sample_test
)
from core.load_data.normalizing import transNorm
from MODELS.loss_functions.get_loss_function import get_lossFun
def train_NN_model(args, model, optim):
    # preparing training dataset
    dataset_dictionary = loadData(args, trange=args["t_train"])
    ### normalizing
    # creating the stats for normalization of NN inputs
    init_norm_stats(args, dataset_dictionary["x_NN"], dataset_dictionary["c_NN"], dataset_dictionary["obs"])
    # normalize
    x_NN_scaled = transNorm(args, dataset_dictionary["x_NN"], varLst=args["varT_NN"], toNorm=True)
    obs_scaled = transNorm(args, dataset_dictionary["obs"], varLst=args["target"], toNorm=True)
    c_NN_scaled = transNorm(args, dataset_dictionary["c_NN"], varLst=args["varC_NN"], toNorm=True)
    c_NN_scaled = np.repeat(np.expand_dims(c_NN_scaled, 0), x_NN_scaled.shape[0], axis=0)
    del dataset_dictionary["x_NN"],   # no need the real values anymore
    dataset_dictionary["inputs_NN_scaled"] = np.concatenate((x_NN_scaled, c_NN_scaled), axis=2)
    dataset_dictionary["obs_scaled"] = obs_scaled
    del x_NN_scaled, c_NN_scaled   # we just need "inputs_NN_model" which is a combination of these two
    ### defining the loss function
    lossFun = get_lossFun(args, dataset_dictionary["obs"])  # obs is needed for certain loss functions, not all of them
    if torch.cuda.is_available():
        model = model.to(args["device"])
        lossFun = lossFun.to(args["device"])
        torch.backends.cudnn.deterministic = True
        CUDA_LAUNCH_BLOCKING = 1

    ngrid_train, nIterEp, nt, batchSize, D_N_P_new = No_iter_nt_ngrid("t_train", args,
                                                                      dataset_dictionary["inputs_NN_scaled"])
    model.zero_grad()
    model.train()
    # training
    for epoch in range(1, args["EPOCHS"] + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(1, nIterEp + 1):
            dataset_dictionary_sample = take_sample_train(args, dataset_dictionary, ngrid_train, nt, batchSize,
                                                          D_N_P_new)
            # Batch running of the differentiable model
            out_model = model(dataset_dictionary_sample)

            # loss function
            loss = lossFun(args, out_model,
                           dataset_dictionary_sample["obs_scaled"],
                           igrid=dataset_dictionary_sample["iGrid"])
            loss.backward()  # retain_graph=True
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
            if (iIter % 1 == 0) or (iIter == nIterEp):
                print(iIter, " from ", nIterEp, " in the ", epoch,
                      "th epoch, and Loss is ", loss.item())
        lossEp = lossEp / nIterEp
        logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory".format(
            epoch, lossEp, time.time() - t0,
            int(torch.cuda.memory_allocated(device=args["device"]) * 0.001))
        print(logStr)

        if epoch % args["saveEpoch"] == 0:
            # save model
            modelFile = os.path.join(args["out_dir"], "model_Ep" + str(epoch) + ".pt")
            torch.save(model, modelFile)
        if epoch == args["EPOCHS"]:
            print("last epoch")
    print("Training ended")


def test_NN_model(args, model):
    warm_up = args["warm_up"]
    # nmul = args["nmul"]
    model.eval()
    # read data for test time range
    dataset_dictionary = loadData(args, trange=args["t_test"])
    #np.save(os.path.join(args["out_dir"], "x.npy"), dataset_dictionary["x_NN"])  # saves with the overlap in the beginning
    # normalizing
    x_NN_scaled = transNorm(args, dataset_dictionary["x_NN"], varLst=args["varT_NN"], toNorm=True)
    c_NN_scaled = transNorm(args, dataset_dictionary["c_NN"], varLst=args["varC_NN"], toNorm=True)
    c_NN_scaled = np.repeat(np.expand_dims(c_NN_scaled, 0), x_NN_scaled.shape[0], axis=0)
    dataset_dictionary["inputs_NN_scaled"] = np.concatenate((x_NN_scaled, c_NN_scaled), axis=2)
    del x_NN_scaled, dataset_dictionary["x_NN"]
    # converting the numpy arrays to torch tensors:
    for key in dataset_dictionary.keys():
        if type(dataset_dictionary[key]) == np.ndarray:  # to avoid int or others to be saved
            dataset_dictionary[key] = torch.from_numpy(dataset_dictionary[key]).float()

    # args_mod = args.copy()
    args["batch_size"] = args["no_basins"]
    nt, ngrid, nx = dataset_dictionary["inputs_NN_scaled"].shape
    rho = args["rho"]

    batch_size = args["batch_size"]
    iS = np.arange(0, ngrid, batch_size)
    iE = np.append(iS[1:], ngrid)
    list_out_model = []
    for i in range(0, len(iS)):
        dataset_dictionary_sample = take_sample_test(args, dataset_dictionary, iS[i], iE[i])
        out_model = model(dataset_dictionary_sample)
        # Convert all tensors in the dictionary to CPU
        out_model_cpu = out_model.cpu().detach()
        out_model_cpu_real = transNorm(args, out_model_cpu.numpy(), varLst=args["target"], toNorm=False)
        # out_diff_model_cpu = tuple(outs.cpu().detach() for outs in out_diff_model)
        list_out_model.append(torch.tensor(out_model_cpu_real))

    # getting rid of warm-up period in observation dataset
    y_obs = dataset_dictionary["obs"]

    save_outputs(args, list_out_model, y_obs, calculate_metrics=True)
    torch.cuda.empty_cache()
    print("Testing ended")
def save_outputs(args, list_out_model, y_obs, calculate_metrics=True):

    if len(list_out_model[0].shape) == 3:
        dim = 1
    else:
        dim = 0
    concatenated_tensor = torch.cat(list_out_model, dim=dim)
    file_name = "NN_sim" + ".npy"
    np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), concatenated_tensor.numpy())

    # Reading flow observation
    for var in args["target"]:
        item_obs = y_obs[:, :, args["target"].index(var)]
        file_name = var + ".npy"
        np.save(os.path.join(args["out_dir"], args["testing_dir"], file_name), item_obs)

    if calculate_metrics == True:
        predLst = list()
        obsLst = list()
        name_list = []
        if "00060_Mean" in args["target"]:
            flow_sim = concatenated_tensor[:, :, args["target"].index("00060_Mean")]
            flow_obs = y_obs[:, :, args["target"].index("00060_Mean")]
            predLst.append(flow_sim.numpy())
            obsLst.append(np.expand_dims(flow_obs, 2))
            name_list.append("flow")
        if "00010_Mean" in args["target"]:
            temp_sim = concatenated_tensor[:, :, args["target"].index("00010_Mean")]
            temp_obs = y_obs[:, :, args["target"].index("00010_Mean")]
            predLst.append(temp_sim.numpy())
            obsLst.append(np.expand_dims(temp_obs, 2))
            name_list.append("temp")


        # we need to swap axes here to have [basin, days]
        statDictLst = [
            stat.statError(np.swapaxes(x.squeeze(), 1, 0), np.swapaxes(y.squeeze(), 1, 0))
            for (x, y) in zip(predLst, obsLst)
        ]
        ### save this file
        # median and STD calculation
        for st, name in zip(statDictLst, name_list):
            count = 0
            mdstd = np.zeros([len(st), 3])
            for key in st.keys():
                median = np.nanmedian(st[key])  # abs(i)
                STD = np.nanstd(st[key])  # abs(i)
                mean = np.nanmean(st[key])  # abs(i)
                k = np.array([[median, STD, mean]])
                mdstd[count] = k
                count = count + 1
            mdstd = pd.DataFrame(
                mdstd, index=st.keys(), columns=["median", "STD", "mean"]
            )
            mdstd.to_csv((os.path.join(args["out_dir"], args["testing_dir"], "mdstd_" + name + ".csv")))

            # Show boxplots of the results
            plt.rcParams["font.size"] = 14
            keyLst = ["Bias", "RMSE", "KGE", "NSE", "Corr"]
            dataBox = list()
            for iS in range(len(keyLst)):
                statStr = keyLst[iS]
                temp = list()
                # for k in range(len(st)):
                data = st[statStr]
                data = data[~np.isnan(data)]
                temp.append(data)
                dataBox.append(temp)
            labelname = [
                "NN model"
            ]  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']

            xlabel = ["Bias ($\mathregular{deg}$C)", "RMSE", "ubRMSE", "NSE", "Corr"]
            fig = plot.plotBoxFig(
                dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8)
            )
            fig.patch.set_facecolor("white")
            boxPlotName = "PGML"
            fig.suptitle(boxPlotName, fontsize=12)
            plt.rcParams["font.size"] = 12
            plt.savefig(
                os.path.join(args["out_dir"], args["testing_dir"], "Box_" + name + ".png")
            )  # , dpi=500
            # fig.show()
            plt.close()










