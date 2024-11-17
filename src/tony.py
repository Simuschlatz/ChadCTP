"""
Credit to Tony Wang https://scholar.google.com/citations?user=TBgVqq4AAAAJ&hl=en
"""
import sys
import os
from qtpy import QtWidgets
from pathlib import Path
from pyvistaqt import QtInteractor, MainWindow
import numpy as np

# Ensure QT_API is set before importing other Qt modules
os.environ["QT_API"] = "pyqt5"

# Path to the volume file
PATH_TO_FILE = Path("volume.npy")

class MyMainWindow(MainWindow):

    def __init__(self, volume):
        super().__init__()

        self.volume = volume
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # PyVista interactor
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        self.show()
        self.add_volume()

    def add_volume(self):
        """ Add a volume to the pyqt frame """
        if self.volume is not None:
            print("Adding volume to plotter")
            offset_volume = create_offset_volume(self.volume)
            opacity = [0, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0]
            self.plotter.add_volume(offset_volume, opacity=opacity, cmap='jet')
            self.plotter.reset_camera()
        else:
            print("No volume data to display")

def create_offset_volume(volume, offset=5):
    """ Create an offset volume with space between each slice """
    slices, height, width = volume.shape
    new_volume = np.zeros((slices * offset, height, width))

    for i in range(slices):
        new_volume[i * offset, :, :] = volume[i, :, :]

    return new_volume

def load_volume(file_path):
    try:
        volume = np.load(file_path)
        print(f"Volume loaded with shape: {volume.shape}")
        return volume
    except Exception as e:
        print(f"Error loading volume: {e}")
        return None

def main():
    volume = load_volume(PATH_TO_FILE)
    if volume is not None:
        app = QtWidgets.QApplication(sys.argv)
        window = MyMainWindow(volume=volume)
        window.show()  # Ensure the window is shown
        app.exec_()
    else:
        print("Failed to load volume data")

if __name__ == "__main__":
    main()