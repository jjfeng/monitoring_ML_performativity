import pickle
import copy

import numpy as np
import pandas as pd


class MonitoringHistory:
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.obs_statistics = []
        self.chart_statistics = []
        self.control_limits = []
        self.batch_sizes = []
        self.num_missings = []
        self.time_idxs = []
        self.candidate_changeponts = []

        self.mdl_history = []

    def update(
        self,
        obs_stat,
        chart_stat,
        control_lim,
        batch_size: int,
        num_missing: int,
        time_idx: int,
        candidate_changepoint: int,
        mdl = None,
    ):
        self.obs_statistics.append(obs_stat)
        self.chart_statistics.append(chart_stat)
        self.control_limits.append(control_lim)
        self.batch_sizes.append(batch_size)
        self.num_missings.append(num_missing)
        self.time_idxs.append(time_idx)
        self.candidate_changeponts.append(candidate_changepoint)
        if mdl is not None:
            self.mdl_history.append({
                "time": time_idx,
                "mdl": copy.deepcopy(mdl)
            })

    @property
    def has_alert(self):
        control_lims = np.array(self.control_limits)
        upper_cross = np.array(self.chart_statistics) > control_lims[:, 1]
        lower_cross = np.array(self.chart_statistics) < control_lims[:, 0]
        return np.any(np.bitwise_or(upper_cross, lower_cross))

    @property
    def alert_time(self):
        control_lims = np.array(self.control_limits)
        upper_cross = np.array(self.chart_statistics) > control_lims[:, 1]
        lower_cross = np.array(self.chart_statistics) < control_lims[:, 0]
        did_alarms = np.bitwise_or(upper_cross, lower_cross)
        alert_idx = np.where(did_alarms)[0]
        if len(alert_idx):
            return self.time_idxs[alert_idx[0]]
        else:
            return alert_idx

    def write_mdls(self, file_name: str):
        with open(file_name, "wb") as f:
            pickle.dump(self.mdl_history, f)

    def write_to_csv(self, csv_file: str):
        control_lims = np.array(self.control_limits)
        obs_statistics = np.array(self.obs_statistics)
        chart_dat = pd.DataFrame(
            {
                "t": self.time_idxs,
                "num_obs": self.batch_sizes,
                "missings": self.num_missings,
                "not_missings": np.array(self.batch_sizes)
                - np.array(self.num_missings),
                "chart_obs": self.chart_statistics,
                "lower_thres": control_lims[:, 0],
                "upper_thres": control_lims[:, 1],
                "candidate_changepoints": self.candidate_changeponts,
            }
        )
        for i in range(obs_statistics.shape[1]):
            chart_dat["batch_mean%d" % i] = obs_statistics[:, i]
        chart_dat["method"] = self.method_name
        chart_dat.to_csv(csv_file, index=False)
