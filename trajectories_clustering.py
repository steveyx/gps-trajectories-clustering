import numpy as np
import pandas as pd

from rdp import rdp, rdp_with_index
import utm
from sklearn.cluster import DBSCAN
import time
import similaritymeasures


class TrajectoryClustering:
    df_move = pd.read_csv("data/input/move_days.csv")
    df_cluster = pd.read_csv("data/input/clusters.csv")
    df_logs = pd.read_csv("data/input/gps_logs.csv")

    @classmethod
    def get_trajectories(cls, cluster_ini, cluster_end):
        _df_m = cls.df_move
        _df_l = cls.df_logs
        _f1 = _df_m["cluster_ini"] == cluster_ini
        _f2 = _df_m["cluster_end"] == cluster_end
        _moves = cls.df_move.loc[_f1 & _f2]
        _trajectories = []
        for idx, _m in _moves.iterrows():
            _day_num = _m["day"]
            _veh = _m["vehicle_id"]
            _t = _m["ts_end"]
            _c1 = _df_l["day"] == _day_num
            _c2 = _df_l["vehicle_id"] == _veh
            _c4 = _df_l["lat"] != _df_l["lat"].shift(1)
            _c5 = _df_l["lon"] != _df_l["lon"].shift(1)
            _cols = ["day", "vehicle_id", "trip", "lat", "ts", "lon"]
            _cols = ["lat", "lon"]
            _df_v_l = _df_l.loc[_c1 & _c2 & _c4 & _c5, _cols]
            if not _df_v_l.empty:
                _trajectories.append(_df_v_l[_cols].values)
        return _trajectories

    @staticmethod
    def convert_lat_lon_to_xy(trajectories):
        _traj_xy = []
        for i, t in enumerate(trajectories):
            _xyzz = np.array([list(utm.from_latlon(ll[0], ll[1])[:2]) for ll in t])
            _traj_xy.append(_xyzz)
        return _traj_xy

    @classmethod
    def compute_distance_matrix(cls, trajectories, method="Frechet"):
        """
        :param method: "Frechet" or "Area"
        """
        n = len(trajectories)
        dist_m = np.zeros((n, n))
        for i in range(n - 1):
            p = trajectories[i]
            for j in range(i + 1, n):
                q = trajectories[j]
                if method == "Frechet":
                    dist_m[i, j] = similaritymeasures.frechet_dist(p, q)
                else:
                    dist_m[i, j] = similaritymeasures.area_between_two_curves(p, q)
                dist_m[j, i] = dist_m[i, j]
        return dist_m

    @classmethod
    def reduce_polyline_points_by_rdp(cls, polyline, epsilon=10, return_indices=False):
        """
        :param polyline:
        :param epsilon: unit m for Frechet distance, m^2 for Area
        :param return_indices: boolean
        """
        point_list = polyline.tolist()
        if return_indices:
            _p_indices = list(range(len(point_list)))
            points, indices = rdp_with_index(point_list, _p_indices, epsilon=epsilon)
            return np.array(points), np.array(indices)
        else:
            points = rdp(point_list, epsilon=epsilon)
            return np.array(points)

    @classmethod
    def clustering_by_dbscan(cls, distance_matrix, eps=1000):
        """
        :param eps: unit m for Frechet distance, m^2 for Area
        """
        cl = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        cl.fit(distance_matrix)
        return cl.labels_

    @classmethod
    def load_cluster(cls, cluster_id):
        _df = cls.df_cluster
        _locations = _df.loc[_df["cluster_id"] == cluster_id, ["lat", "lon"]].values
        return _locations[0]


if __name__ == "__main__":
    clusters = [[1, 184], [144, 11], [1, 173], [11, 22], [29, 184], [29, 185]]
    cluster_start, cluster_end = clusters[1]

    trajectories = TrajectoryClustering.get_trajectories(cluster_start, cluster_end)
    print("number of trajectories: {}".format(len(trajectories)))
    trajectories_xy = TrajectoryClustering.convert_lat_lon_to_xy(trajectories)
    t0 = time.time()
    dist_mat = TrajectoryClustering.compute_distance_matrix(trajectories_xy)
    t1 = time.time()
    print("distance matrix without rdp completes in {} seconds".format(t1 - t0))
    t0 = time.time()
    trajectories_reduced = [TrajectoryClustering.reduce_polyline_points_by_rdp(p) for p in trajectories_xy]
    dist_mat_reduced = TrajectoryClustering.compute_distance_matrix(trajectories_reduced)
    t1 = time.time()
    print("distance matrix with rdp completes in {} seconds.".format(t1 - t0))
    data = [[len(t), len(trajectories_reduced[i])] for i, t in enumerate(trajectories)]
    df = pd.DataFrame(data=data, columns=["original data points", "after RDP"])
    df.index.name = "traj #"
    df.to_csv("data/output/RDP_data_points.csv")

