import sys
import os
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from qtpy.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QLineEdit, QPushButton, QMessageBox, QFileDialog, QComboBox
)
from dustmaps.config import config
import dustmaps.sfd
import dustmaps.bayestar
import dustmaps.planck
import dustmaps.decaps

from healpy.projector import GnomonicProj

config['data_dir'] = '/Users/agoodman/Library/CloudStorage/GoogleDrive-agoodman@cfa.harvard.edu/My Drive/Milky Way Takeover 2020/MilkyWay3D.ORG/Dustmaps Experiment'

MAPS = {
    'SFD': (dustmaps.sfd.SFDQuery, dustmaps.sfd.fetch),
    'Bayestar': (dustmaps.bayestar.BayestarQuery, dustmaps.bayestar.fetch),
    'Planck': (dustmaps.planck.PlanckQuery, dustmaps.planck.fetch),
    'DECaPS': (dustmaps.decaps.DECaPSQuery, dustmaps.decaps.fetch)
}

FORMATS = {
    'SFD': ["WCS FITS", "CSV"],
    'Bayestar': ["WCS FITS", "CSV"],
    'Planck': ["WCS FITS", "CSV"],
    'DECaPS': ["WCS FITS", "CSV"]
}

class DustApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dustmap Loader")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.map_selector = QComboBox()
        self.map_selector.addItems(MAPS.keys())
        self.map_selector.currentIndexChanged.connect(self.update_format_options)
        layout.addWidget(self.map_selector)

        self.l_input = QLineEdit()
        self.l_input.setPlaceholderText("Galactic Longitude Center (l deg)")
        layout.addWidget(self.l_input)

        self.b_input = QLineEdit()
        self.b_input.setPlaceholderText("Galactic Latitude Center (b deg)")
        layout.addWidget(self.b_input)

        self.radius_input = QLineEdit()
        self.radius_input.setPlaceholderText("Radius (deg)")
        layout.addWidget(self.radius_input)

        self.format_selector = QComboBox()
        layout.addWidget(self.format_selector)

        self.download_btn = QPushButton("Download WCS Map")
        self.download_btn.clicked.connect(self.download_map)
        layout.addWidget(self.download_btn)

        self.setLayout(layout)
        self.update_format_options()

    def update_format_options(self):
        self.format_selector.clear()
        self.format_selector.addItems(FORMATS[self.map_selector.currentText()])

    def download_map(self):
        try:
            map_name = self.map_selector.currentText()
            QueryClass, fetch_map = MAPS[map_name]

            if not os.path.exists(os.path.join(config['data_dir'], map_name.lower())):
                fetch_map()

            try:
                l_center = float(self.l_input.text())
                b_center = float(self.b_input.text())
                radius = float(self.radius_input.text())
            except ValueError:
                QMessageBox.critical(self, "Input Error", "Please enter valid numbers for l, b, and radius.")
                return

            xsize = ysize = 400
            reso_arcmin = (radius * 60 * 2) / xsize
            proj = GnomonicProj(rot=(l_center, b_center), xsize=xsize, ysize=ysize, reso=reso_arcmin, coord='G')

            filename, _ = QFileDialog.getSaveFileName(self, "Save Map", f"{map_name}_output", "FITS (*.fits);;CSV (*.csv)")
            if not filename:
                return

            xpix, ypix = np.meshgrid(np.arange(xsize), np.arange(ysize))
            vec = proj.xy2vec(xpix.flatten(), ypix.flatten())
            coords = SkyCoord(vec[0], vec[1], vec[2],
                              representation_type='cartesian', frame='galactic')

            dustmap = QueryClass()
            ebv = dustmap(coords)

            if self.format_selector.currentText() == "CSV":
                np.savetxt(filename, np.vstack((coords.l.deg, coords.b.deg, ebv)).T,
                           delimiter=',', header='l,b,E(B-V)', comments='')
            else:
                image = ebv.reshape((ysize, xsize))
                header = proj.wcs.to_header()
                fits.writeto(filename, image, header, overwrite=True)
                QMessageBox.information(self, "Success", f"Map saved to: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DustApp()
    window.show()
    sys.exit(app.exec())
