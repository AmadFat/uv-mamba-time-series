import click
import numpy as np
import math
from PyQt6 import QtWidgets
import pyqtgraph.opengl as gl

def load_train_data(root: str):
    """Load longitude and latitude data from LMDB database"""
    import lmdb
    import json
    from pathlib import Path

    road_paths = []
    env_path = str(Path(root) / "train.lmdb")
    env = lmdb.Environment(env_path, readahead=True)
    txn = env.begin(write=False)
    keys = json.loads(txn.get(b"__keys__"))
    for k in keys:
        data = json.loads(txn.get(k.encode()))
        longitude = [l / 10 for l in data["longitude"]]
        latitude = [l / 10 for l in data["latitude"]]
        road_paths.append((longitude, latitude))
    return road_paths

def load_test_data(root: str):
    """Load test data including velocity and direction information"""
    import lmdb
    import json
    from pathlib import Path

    road_paths = []
    env_path = str(Path(root) / "test.lmdb")
    env = lmdb.Environment(env_path, readahead=True)
    txn = env.begin(write=False)
    keys = json.loads(txn.get(b"__keys__"))
    for k in keys:
        data = json.loads(txn.get(k.encode()))
        longitude = data["longitude"]
        latitude = data["latitude"]
        velocity  = data["velocity"]
        direction = data["direction"]
        road_paths.append((longitude, latitude, velocity, direction))
    return road_paths

class EarthVisualizer(QtWidgets.QWidget):
    def __init__(self, paths, mode):
        super().__init__()
        self.setWindowTitle("3D Earth Visualizer")
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 200
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        self.paths = paths
        self.mode = mode
        self._init_earth()
        self._draw_paths()

    def _init_earth(self):
        earth_radius = 6371/100.0
        # Lower row/col count for faster rendering
        mesh = gl.MeshData.sphere(rows=20, cols=20, radius=earth_radius)
        earth = gl.GLMeshItem(meshdata=mesh, smooth=True,
                              color=(0.5,0.5,1,1), shader='shaded',
                              glOptions='opaque')
        self.view.addItem(earth)

    def _latlon_to_xyz(self, lat, lon, radius):
        # convert degrees to radians
        phi = math.radians(90.0 - lat)
        theta = math.radians(lon + 180.0)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        return np.array([x, y, z])

    def _draw_paths(self):
        earth_radius = 6371/100.0
        
        # Batch points for each path type
        train_points = []
        test_points = []
        arrow_starts = []
        arrow_ends = []
        
        for path in self.paths:
            if self.mode == 'train':
                lon, lat = path
                pts = np.array([
                    self._latlon_to_xyz(lat[i], lon[i], earth_radius)
                    for i in range(len(lat))
                ])
                train_points.append(pts)
            else:
                lon, lat, vel, dirc = path
                pts = np.array([
                    self._latlon_to_xyz(lat[i], lon[i], earth_radius)
                    for i in range(len(lat))
                ])
                test_points.append(pts)
                
                # Batch arrow data
                for i, v in enumerate(vel):
                    start = pts[i]
                    d_rad = math.radians(dirc[i])
                    # local north vector
                    nvec = self._latlon_to_xyz(lat[i]+1e-5, lon[i], earth_radius) - start
                    nvec = nvec / np.linalg.norm(nvec)
                    # local east vector
                    evec = self._latlon_to_xyz(lat[i], lon[i]+1e-5, earth_radius) - start
                    evec = evec / np.linalg.norm(evec)
                    dir_vec = math.cos(d_rad)*nvec + math.sin(d_rad)*evec
                    end = start + dir_vec * v / 4
                    arrow_starts.append(start)
                    arrow_ends.append(end)
        
        # Draw train paths with points
        if self.mode == 'train' and train_points:
            all_points = np.vstack(train_points)
            scatter = gl.GLScatterPlotItem(pos=all_points, color=(1,0,0,1), 
                                         size=2, pxMode=True)
            self.view.addItem(scatter)
            
            # Also add lines with reduced antialiasing for path clarity
            for pts in train_points:
                line = gl.GLLinePlotItem(pos=pts, color=(1,0,0,0.5),
                                       width=1, antialias=False)
                self.view.addItem(line)
        
        # Draw test paths
        if self.mode == 'test' and test_points:
            all_test_points = np.vstack(test_points)
            scatter = gl.GLScatterPlotItem(pos=all_test_points, color=(0,1,0,1), 
                                         size=2, pxMode=True)
            self.view.addItem(scatter)
            
            for pts in test_points:
                line = gl.GLLinePlotItem(pos=pts, color=(0,1,0,0.5),
                                       width=1, antialias=False)
                self.view.addItem(line)
            
            # Draw arrows as a single batch
            if arrow_starts and arrow_ends:
                arrows = np.empty((len(arrow_starts)*2, 3))
                arrows[0::2] = arrow_starts
                arrows[1::2] = arrow_ends
                arrow_item = gl.GLLinePlotItem(pos=arrows, color=(1,1,0,0.7), 
                                           width=0.5, mode='lines', antialias=False)
                self.view.addItem(arrow_item)
        
        # highlight specific point
        lat0, lon0 = 19.17, 110.75
        mark = self._latlon_to_xyz(lat0, lon0, earth_radius)
        mmesh = gl.MeshData.sphere(rows=10, cols=10, radius=250/100.0)
        marker = gl.GLMeshItem(meshdata=mmesh, smooth=False,
                             color=(1,0,1,1), shader='shaded')
        marker.translate(*mark)
        self.view.addItem(marker)

@click.command()
@click.argument('root', default="/home/af/Data/mamba-time-series")
@click.option('--mode', type=click.Choice(['train','test']), default='test')
def main(root, mode):
    paths = load_train_data(root) if mode=='train' else load_test_data(root)
    app = QtWidgets.QApplication([])
    viz = EarthVisualizer(paths, mode)
    viz.show()
    app.exec()

if __name__ == '__main__':
    main()
