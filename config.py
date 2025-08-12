from io import BytesIO
import os
import sys
import traceback

from dustmaps.config import config
import dustmaps.sfd
import dustmaps.bayestar
import dustmaps.planck
import dustmaps.decaps

from glue.config import importer
from glue.core import Data
from glue.core.data_factories.fits import fits_reader

from astropy.io.fits import HDUList, PrimaryHDU, writeto
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from healpy.projector import GnomonicProj
import numpy as np
from qtpy.QtWidgets import (
    QApplication, QWidget, QFormLayout, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, QComboBox, QDialog
)

config["data_dir"] = "/home/jon/dev/glue/data"

MAPS = {
    'SFD': (dustmaps.sfd.SFDQuery, dustmaps.sfd.fetch),
    'Bayestar': (dustmaps.bayestar.BayestarQuery, dustmaps.bayestar.fetch),
    'Planck': (dustmaps.planck.PlanckQuery, dustmaps.planck.fetch),
    'DECaPS': (dustmaps.decaps.DECaPSQuery, dustmaps.decaps.fetch)
}

class DustmapLoaderWidget(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dustmap UI Mockup")
        self.setup_ui()
        self.data = None

    def setup_ui(self):
        layout = QVBoxLayout()

        # Map selector
        self.map_selector = QComboBox()
        self.map_selector.addItems(["SFD", "Planck", "Bayestar", "DECaPS"])
        self.map_selector.currentIndexChanged.connect(self.update_options_panel)
        layout.addWidget(QLabel("Select Map:"))
        layout.addWidget(self.map_selector)

        # Region controls
        layout.addWidget(QLabel("Center (l, b in degrees) and Radius (deg):"))
        region_layout = QHBoxLayout()
        self.l_input = QLineEdit(); self.l_input.setPlaceholderText("l")
        self.b_input = QLineEdit(); self.b_input.setPlaceholderText("b")
        self.r_input = QLineEdit(); self.r_input.setPlaceholderText("radius")
        for w in [self.l_input, self.b_input, self.r_input]:
            region_layout.addWidget(w)
        layout.addLayout(region_layout)

        # Just for testing
        self.l_input.setText("0")
        self.b_input.setText("0")
        self.r_input.setText("10")

        # Dynamic map-specific options
        self.options_group = QGroupBox("Map-Specific Options")
        self.options_layout = QFormLayout()
        self.options_group.setLayout(self.options_layout)
        layout.addWidget(self.options_group)

        # Dummy button
        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.download_map)
        layout.addWidget(self.download_btn)

        self.setLayout(layout)
        self.update_options_panel()

    def update_options_panel(self):
        # Clear current options
        while self.options_layout.rowCount():
            self.options_layout.removeRow(0)

        map_name = self.map_selector.currentText()
        if map_name == "Bayestar":
            self.add_distance_range_fields()
        elif map_name == "DECaPS":
            self.add_distance_range_fields()
            self.res_input = QLineEdit()
            self.res_input.setPlaceholderText("Distance step (pc)")
            self.options_layout.addRow("Step:", self.res_input)

    def add_distance_range_fields(self):
        self.dmin_input = QLineEdit(); self.dmin_input.setPlaceholderText("e.g. 100")
        self.dmax_input = QLineEdit(); self.dmax_input.setPlaceholderText("e.g. 5000")
        self.options_layout.addRow("Min Distance (pc):", self.dmin_input)
        self.options_layout.addRow("Max Distance (pc):", self.dmax_input)

    def download_map(self):
        try:
            map_name = self.map_selector.currentText()
            QueryClass, fetch_map = MAPS[map_name]

            if not os.path.exists(os.path.join(config['data_dir'], map_name.lower())):
                fetch_map()

            try:
                l_center = float(self.l_input.text())
                b_center = float(self.b_input.text())
                r = float(self.r_input.text())
            except ValueError:
                QMessageBox.critical(self, "Input Error", "Please enter valid numbers for l, b, and radius.")
                return

            center = SkyCoord(l=l_center * u.deg, b=b_center * u.deg, frame='galactic')
            radius = r * u.deg
            npts = 1000
            
            # Offsets are in degrees here
            lon_offsets = np.linspace(-radius, radius, npts)
            lat_offsets = np.linspace(-radius, radius, npts)
            lon_grid, lat_grid = np.meshgrid(lon_offsets, lat_offsets)
            
            offset_frame = SkyOffsetFrame(origin=center)
            
            grid_in_offset_frame = SkyCoord(
                lon=lon_grid,
                lat=lat_grid,
                distance=center.distance,
                frame=offset_frame
            )
            
            coords = grid_in_offset_frame.transform_to('galactic')

            dustmap = QueryClass()
            ebv = dustmap(coords)
            self.data = [Data(label=map_name, values=ebv)]
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            traceback.print_exc()


@importer("Import dustmaps data")
def dustmaps_importer():
    dialog = DustmapLoaderWidget()
    dialog.exec()
    if dialog.data is not None:
        return dialog.data
    else:
        return []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DustmapLoaderWidget()
    win.show()
    sys.exit(app.exec())
