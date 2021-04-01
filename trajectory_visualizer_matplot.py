import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from trajectories_clustering import TrajectoryClustering


class TrajectoryVisualizerMatplot:
    select_colors = ['darkgreen', 'red', 'purple', 'darkblue', 'orange', 'cyan', 'gray',
                     'darkred', 'darkgreen', 'darkorange', 'pink', 'greenyellow', 'skyblue', 'black',
                     'forestgreen', 'deeppink', 'violet', 'lightblue', 'steelblue', 'yellowgreen',
                     'seagreen', 'blueviolet', 'forestgreen', 'yellow', 'lightgreen']
    markers = ['H', 's', 'd', 'p', '*', 'X', 'P', 'D', 'h', '8', 'v', '^', '<', '>', 'o']

    def __init__(self, map_loc='map/', dpi=150, subplots=(2, 3)):
        roads = gpd.GeoDataFrame.from_file(map_loc + 'ann_map.shp')
        rows, cols = subplots
        self.fig, self.axes = plt.subplots(rows, cols, dpi=dpi, figsize=(6.4, 3.6))
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.05, right=0.95, hspace=0.02)
        if rows * cols > 1:
            self.axes = self.axes.flatten()
        else:
            self.axes = [self.axes]
        for ax in self.axes:
            ax.set_aspect('equal')
            ax.grid()
            roads.plot(ax=ax, color='gray', edgecolor='gray', linewidth=.1)
            ax.set_xlim(103.6, 104.04)
            ax.set_ylim(1.22, 1.475)
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()

    @staticmethod
    def plot_show():
        plt.show()

    def zoom_fit(self, all_logs, margin=0.002):
        latlon_all = np.concatenate([trip for group in all_logs for trip in group], axis=0)
        lon_delta = 0.03
        min_lat, max_lat = latlon_all[:, 0].min(), latlon_all[:, 0].max()
        min_lon, max_lon = latlon_all[:, 1].min() - lon_delta, latlon_all[:, 1].max() + lon_delta
        for ax in self.axes:
            ax.set_xlim(min_lon - margin, max_lon + margin)
            ax.set_ylim(min_lat - margin, max_lat + margin)

    def plot_trajectories(self, trajectories, clusters):
        for i, _trip_clu in enumerate(trajectories):
            ax = self.axes[i]
            _start, _end = clusters[i][0], clusters[i][1]
            for trip in _trip_clu:
                ax.plot(trip[:, 1], trip[:, 0], color=self.select_colors[i % 25])
            _start_loc = TrajectoryClustering.load_cluster(_start)
            _end_loc = TrajectoryClustering.load_cluster(_end)
            ax.text(_start_loc[1], _start_loc[0], "start", fontsize=8, color='black',
                    horizontalalignment='left', verticalalignment='bottom')
            ax.text(_end_loc[1], _end_loc[0], "end", fontsize=8, color='black',
                    horizontalalignment='left', verticalalignment='bottom')
            ax.scatter([_start_loc[1], _end_loc[1]], [_start_loc[0], _end_loc[0]])

    def plot_clustered_trajectories(self, trajectories, start_end_clusters, trip_labels):
        for i, _trip_clu in enumerate(trajectories):
            ax = self.axes[i]
            _labels = trip_labels[i]
            for j_trip, trip in enumerate(_trip_clu):
                ax.plot(trip[:, 1], trip[:, 0], color=self.select_colors[_labels[j_trip] % 25])
            _start, _end = start_end_clusters[i][0], start_end_clusters[i][1]
            _start_loc = TrajectoryClustering.load_cluster(_start)
            _end_loc = TrajectoryClustering.load_cluster(_end)
            ax.text(_start_loc[1], _start_loc[0], "start", fontsize=8, color='black',
                    horizontalalignment='left', verticalalignment='bottom')
            ax.text(_end_loc[1], _end_loc[0], "end", fontsize=8, color='black',
                    horizontalalignment='left', verticalalignment='bottom')
            ax.scatter([_start_loc[1], _end_loc[1]], [_start_loc[0], _end_loc[0]])


if __name__ == '__main__':
    clusters = [[1, 184], [1, 173], [11, 22], [29, 184], [144, 11], [29, 185]]
    clusters = clusters[:]
    all_trips_logs = []
    for cluster_start, cluster_end in clusters:
        g = TrajectoryClustering.get_trajectories(cluster_start, cluster_end)
        all_trips_logs.append(g)
    mv = TrajectoryVisualizerMatplot(subplots=(2, 3))
    mv.plot_trajectories(all_trips_logs, clusters)
    mv.zoom_fit(all_trips_logs)
    mv.plot_show()
