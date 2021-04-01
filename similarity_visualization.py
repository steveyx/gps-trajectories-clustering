from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize
from trajectories_clustering import TrajectoryClustering
from trajectory_visualizer_matplot import TrajectoryVisualizerMatplot
import numpy as np
import geopandas as gpd
from frechet_dist import FrechetDist


class PlotSimilarity:

    @staticmethod
    def plot_area_between(traj0, traj1, switch_xy=True, axis=None):
        if switch_xy:
            traj0 = [[lon, lat] for lat, lon in traj0]
            traj1 = [[lon, lat] for lat, lon in traj1]
        _polygon_points = traj0 + traj1[::-1] + traj0[0:1]
        _polygon = Polygon(_polygon_points)
        x, y = _polygon.exterior.xy
        ls = LineString(np.c_[x, y])
        lr = LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        for _polygon in polygonize(mls):
            p = gpd.GeoSeries(_polygon)
            p.plot(ax=axis, color="lightgreen")
        axis.text(_polygon.centroid.x, _polygon.centroid.y, "Area between two trajectories", fontsize=8, color='black',
                  horizontalalignment='center', verticalalignment='bottom')

    @staticmethod
    def plot_frechet_dist(traj_xy0, traj_xy1,
                          traj0, traj1, axis=None):
        f_d, p_i, q_i = FrechetDist.frechetdist_index(traj_xy0, traj_xy1)
        x = [traj0[p_i][1], traj1[q_i][1]]
        y = [traj0[p_i][0], traj1[q_i][0]]
        axis.plot(x, y, label="frechet dist", linestyle=":", color="purple")
        axis.text(np.mean(x), np.mean(y), "Frechet Dist", fontsize=8, color='black',
                  horizontalalignment='center', verticalalignment='bottom')


if __name__ == "__main__":
    clusters = [[1, 184], [144, 11], [1, 173], [11, 22], [29, 184], [29, 185]]
    cluster_ini, cluster_end = clusters[0]

    trajectories = TrajectoryClustering.get_trajectories(cluster_ini, cluster_end)
    trajectories_xy = TrajectoryClustering.convert_lat_lon_to_xy(trajectories)

    trajectories_reduced = [TrajectoryClustering.reduce_polyline_points_by_rdp(p, return_indices=True)
                            for p in trajectories_xy]
    points_indices = [t[1] for t in trajectories_reduced]
    trajectories_reduced = [t[0] for t in trajectories_reduced]
    dist_mat_reduced = TrajectoryClustering.calculate_distance_matrix(trajectories_reduced)
    labels = TrajectoryClustering.clustering_by_dbscan(dist_mat_reduced, eps=1000)
    print(labels)

    clusters = clusters[:1]
    all_trips_logs = [trajectories]
    mv = TrajectoryVisualizerMatplot(subplots=(1, 1))
    mv.plot_clustered_trajectories(all_trips_logs, clusters, [labels])
    mv.zoom_fit(all_trips_logs)
    mv.plot_show()

    c0 = [i for i, l0 in enumerate(labels) if l0 == 1][0]
    c1 = [i for i, l1 in enumerate(labels) if l1 == 0][0]
    f_d, p_i, q_i = FrechetDist.frechetdist_index(trajectories_reduced[c0], trajectories_reduced[c1])

    clusters = clusters[:1]
    all_trips_logs = [[trajectories[c][points_indices[c], :] for c in [c0, c1]]]
    mv = TrajectoryVisualizerMatplot(subplots=(1, 1))
    mv.plot_clustered_trajectories(all_trips_logs, clusters, [[c0, c1]])
    mv.zoom_fit(all_trips_logs)
    PlotSimilarity.plot_area_between(all_trips_logs[0][0], all_trips_logs[0][1], axis=mv.axes[0])
    PlotSimilarity.plot_frechet_dist(trajectories_reduced[c0], trajectories_reduced[c1],
                                     all_trips_logs[0][0], all_trips_logs[0][1], axis=mv.axes[0])
    mv.plot_show()