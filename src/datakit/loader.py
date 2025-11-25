import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd

from ..utils.myplot import figure_grid


USER_DATA_DF = pd.read_csv(os.path.join(Path(__file__).parent.parent.parent, "data/behavior/user_data_sample.csv"))
N_USER = 40

def get_user_data(user_id: int):
    """
    Get data for a specific user by user_id.
    """
    assert 0 <= user_id <= N_USER - 1, f"user_id out of range: [0, {N_USER - 1}]"
    user_data = USER_DATA_DF[USER_DATA_DF["user_id"] == user_id]
    # rt = user_data["reaction_time"].to_numpy()
    # cor = user_data["correct"].to_numpy()
    return user_data




################ DO NOT USE BELOW FUNCTIONS ################

# def _generate_simulated_data():
#     from ..simulator.drift_diffusion import drift_diffusion_simulation, A_RANGE, MU_RANGE, T_ER_RANGE
    
#     df_all = list()
#     param_all = list()
#     for user_id in tqdm(range(40)):
#         a = np.random.uniform(A_RANGE[0], A_RANGE[1])
#         mu = np.random.uniform(MU_RANGE[0], MU_RANGE[1])
#         T_er = np.random.uniform(T_ER_RANGE[0], T_ER_RANGE[1])

#         rt, cor = drift_diffusion_simulation(a, mu, T_er, num_of_simul=80, verbose=False, _assert=False)

#         save_path = os.path.join(Path(__file__).parent.parent.parent, "data/behavior/")
#         os.makedirs(save_path, exist_ok=True)

#         df = pd.DataFrame({"user_id": user_id, "reaction_time": rt, "correct": cor})
#         df_all.append(df)
#         param_all.append({"user_id": user_id, "a": a, "mu": mu, "T_er": T_er})

#     df_all = pd.concat(df_all, ignore_index=True)
#     df_all.to_csv(os.path.join(save_path, f"simul_data_sample.csv"), index=False)

#     param_all = pd.DataFrame(param_all)
#     param_all.to_csv(os.path.join(save_path, f"simul_data_params.csv"), index=False)



# def _process_raw_data(file_path: str, file_name: str):
#     """
#     Process raw data file into desired format.
#     Original data from T1 Collaboration project (GMS)
#     """
#     save_path = os.path.join(Path(__file__).parent.parent.parent, "data/user/processed_data")
#     os.makedirs(save_path, exist_ok=True)

#     data = pd.read_csv(file_path)
#     data = data[data["NumLight"] == 2]
#     data["correct"] = (data["Colored"] == data["Pressed"]).astype(int)

#     del data["NumLight"]
#     del data["Trial"]
#     del data["Colored"]
#     del data["Pressed"]
#     del data["trialStartTime"]
#     del data["PressedTime"]
#     if "Correct" in data.columns:
#         del data["Correct"]

#     data.rename(columns={"ReactionTime": "reaction_time"}, inplace=True)
#     data = data[(data["reaction_time"] <= 0.75) & (data["reaction_time"] >= 0.1)]
#     if not data.empty and np.mean(data["correct"].to_numpy()) >= 0.7 and data.shape[0] >= 60:
#         data.to_csv(os.path.join(save_path, file_name), index=False)


# def _process_all_raw_data():
#     import glob
#     file_list = glob.glob(os.path.join(Path(__file__).parent.parent.parent, "data/user/raw_data/*.csv"))
#     file_list.sort()

#     for i, data_file in enumerate(tqdm(file_list)):
#         _process_raw_data(data_file, f"processed_data_{i+1:04d}.csv")


# def _visualize_data_scatter():
#     """Visualize scatter plot with index numbers for data selection."""
#     rt = list()
#     cor = list()

#     index_list = [326, 917, 608, 609, 1217,
#                   1152, 613, 292, 633, 684,
#                   402, 400, 480, 1214, 776,
#                   96, 1160, 614, 806, 1065,
#                   600, 1132, 141, 631, 1110,
#                   1024, 1144, 513, 981, 162,
#                   999, 429, 758, 950, 1091,
#                   200, 204, 137, 722, 809]
#     import glob
#     file_list = glob.glob(os.path.join(Path(__file__).parent.parent.parent, "data/user/processed_data/*.csv"))  
#     file_list.sort()

#     for i, data_file in enumerate(tqdm(file_list)):
#         D = pd.read_csv(data_file)

#         rt.append(D.reaction_time.to_numpy().mean())
#         cor.append(D.correct.to_numpy().mean())
    
#     rt, cor = np.array(rt), np.array(cor)
    
#     # plt.scatter(rt, cor, s=15, alpha=0.5)
#     plt.scatter(rt[index_list], cor[index_list], s=15, alpha=0.5)
#     # for i, (x, y) in enumerate(zip(rt, cor)):
#         # plt.text(x, y, str(i), fontsize=7, ha='center', va='bottom')
#     plt.xlabel("Mean Reaction Time")
#     plt.ylabel("Mean Correctness")
#     plt.show()


# def _create_user_data_csv():
#     index_list = [326, 917, 608, 609, 1217,
#                   1152, 613, 292, 633, 684,
#                   402, 400, 480, 1214, 776,
#                   96, 1160, 614, 806, 1065,
#                   600, 1132, 141, 631, 1110,
#                   1024, 1144, 513, 981, 162,
#                   999, 429, 758, 950, 1091,
#                   200, 204, 137, 722, 809]

#     import glob
#     file_list = glob.glob(os.path.join(Path(__file__).parent.parent.parent, "data/user/processed_data/*.csv"))  
#     file_list.sort()

#     df_list = list()

#     for ii, i in enumerate(index_list):
#         data_file = file_list[i]
#         df = pd.read_csv(data_file)
#         df["user_id"] = ii
#         df = df[["user_id"] + [col for col in df.columns if col != "user_id"]]
#         df_list.append(df)
    
#     df_all = pd.concat(df_list, ignore_index=True)
#     save_path = os.path.join(Path(__file__).parent.parent.parent, "data/behavior/")
#     os.makedirs(save_path, exist_ok=True)
#     df_all.to_csv(os.path.join(save_path, "user_data_sample.csv"), index=False)




if __name__ == "__main__":
    # _process_all_raw_data()
    # _visualize_data_scatter()
    # _create_user_data_csv()
    # _generate_simulated_data()
    pass