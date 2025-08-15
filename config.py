from dataclasses import dataclass, field, fields
import os
import sys
import traceback
from PyQt5.QtGui import QDoubleValidator, QFont, QIntValidator

from dustmaps.config import config
import dustmaps.sfd
import dustmaps.bayestar
import dustmaps.planck
import dustmaps.decaps

from glue.config import importer
from glue.core import Data

from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
import numpy as np
from qtpy.QtWidgets import (
    QApplication, QFormLayout, QGroupBox, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, QComboBox, QDialog, QFileDialog
)

config["data_dir"] = "/media/jon/Seagate Backup Plus Drive1/dev/glue/data"


# TODO: I don't feel that the dataclass model is really correct as we never instantiate the classes
# But it's also not really a problem at the moment


@dataclass
class BaseOptions:
    l: float = field(metadata={"unit": "deg"})
    b: float = field(metadata={"unit": "deg"})
    r: float = field(metadata={"unit": "deg"})


@dataclass
class Base3DOptions(BaseOptions):
    min_distance: float = field(metadata={"unit": "pc"})
    max_distance: float = field(metadata={"unit": "pc"})


@dataclass
class DECaPSRegionOptions(Base3DOptions):
    distance_step: float = field(metadata={"unit": "pc"})


MAPS = {
    'SFD': (dustmaps.sfd.SFDQuery, dustmaps.sfd.fetch, BaseOptions),
    'Bayestar': (dustmaps.bayestar.BayestarQuery, dustmaps.bayestar.fetch, BaseOptions),
    'Bayestar (Web)': (dustmaps.bayestar.BayestarWebQuery, None, BaseOptions),
    'Planck': (dustmaps.planck.PlanckQuery, dustmaps.planck.fetch, Base3DOptions),
    'DECaPS': (dustmaps.decaps.DECaPSQueryLite, dustmaps.decaps.fetch, DECaPSRegionOptions),
    'DECaPS (Mean)': (lambda: dustmaps.decaps.DECaPSQueryLite(mean_only=True, contiguous=True), lambda: dustmaps.decaps.fetch(mean_only=True), DECaPSRegionOptions)
}



class DustmapLoaderWidget(QDialog):
    def __init__(self):
        super().__init__()
        self._default_font = QFont()
        self._default_font.setBold(False)
        self._bold_font = QFont()
        self._bold_font.setBold(True)
        self.setWindowTitle("Import dustmap data")
        self.setup_ui()
        self.data = None

    def _field_label(self, field):
        text = field.name.replace("_", " ")
        if " " in text:
            text = text.title()
        if (unit := field.metadata.get("unit", None)) is not None:
            text = f"{text} ({unit})"
        return text

    def _widgets_for_field(self, field):
        text = self._field_label(field)
        label = QLabel(text=text)
        input = QLineEdit() 
        t = field.type
        if t is int:
            input.setValidator(QIntValidator())
        elif t is float:
            input.setValidator(QDoubleValidator())
        return label, input

    def _option_widgets(self, info_cls, ignore=None):
        widgets = {}
        for field in fields(info_cls):
            if ignore and field in ignore:
                continue
            widgets[field.name] = self._widgets_for_field(field)
        return widgets

    def _select_dir(self):
        directory = QFileDialog.getExistingDirectory(
                self,
                "Select Directory",
                "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if directory:
            config["data_dir"] = directory
            self.dir_button.setText(directory)

    def setup_ui(self):
        layout = QVBoxLayout()

        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Select directory")
        self.dir_button = QPushButton(config["data_dir"])
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_button)
        self.dir_button.clicked.connect(self._select_dir)
        layout.addLayout(dir_layout)

        # Map selector
        self.map_selector = QComboBox()
        self.map_selector.addItems(MAPS.keys())
        self.map_selector.currentIndexChanged.connect(self.update_options_panel)
        map_label = QLabel("Select Map")
        map_label.setFont(self._bold_font)
        layout.addWidget(map_label)
        layout.addWidget(self.map_selector)

        # Region controls
        region_label = QLabel("Region Options")
        region_label.setFont(self._bold_font)
        layout.addWidget(region_label)
        region_layout = QVBoxLayout()
        widgets = self._option_widgets(BaseOptions)
        self.inputs = { name: w[1] for name, w in widgets.items() }
        self.base_inputs = dict(self.inputs)
        for ws in widgets.values():
            row = QHBoxLayout()
            for w in ws:
                row.addWidget(w)
            region_layout.addLayout(row)
        layout.addLayout(region_layout)

        # Dynamic map-specific options
        self.options_group = QGroupBox("Map-Specific Options")
        self.options_group.setFont(self._bold_font)
        self.options_layout = QFormLayout()
        self.options_group.setLayout(self.options_layout)
        layout.addWidget(self.options_group)

        # Dummy button
        self.import_btn = QPushButton("Import")
        self.import_btn.clicked.connect(self.import_data)
        layout.addWidget(self.import_btn)

        self.setLayout(layout)
        self.update_options_panel()

    def update_options_panel(self):
        # Clear current options
        while self.options_layout.rowCount():
            self.options_layout.removeRow(0)
        self.inputs = dict(self.base_inputs)

        map_name = self.map_selector.currentText()
        options_cls = MAPS[map_name][-1]
        widgets = self._option_widgets(options_cls, ignore=fields(BaseOptions))
        for field_name, ws in widgets.items():
            label, input = ws
            self.inputs[field_name] = input
            input.setFont(self._default_font)
            label.setFont(self._default_font)
            self.options_layout.addRow(label, input)

    def _has_distance(self, options_cls):
        return issubclass(options_cls, Base3DOptions)

    def _input_number(self, key):
        return float(self.inputs[key].text())

    def _input_coords(self, options_cls):
        try:
           l_center = self._input_number("l")
           b_center = self._input_number("b")
           r = self._input_number("r")
        except ValueError:
           QMessageBox.critical(self, "Input Error", "Please enter valid numbers for l, b, and radius.")
           return


        center = SkyCoord(l=l_center * u.deg, b=b_center * u.deg, distance=1*u.pc, frame='galactic')
        radius = r * u.deg

        is_3d = self._has_distance(options_cls)
        npts = 50 if is_3d else 5000
        
        # Offsets are in degrees here
        lon_offsets = np.linspace(-radius, radius, npts)
        lat_offsets = np.linspace(-radius, radius, npts)
        lon_grid, lat_grid = np.meshgrid(lon_offsets, lat_offsets)
        
        offset_frame = SkyOffsetFrame(origin=center)

        if is_3d:
            d_min = self._input_number("min_distance")
            d_max = self._input_number("max_distance")
            if issubclass(options_cls, DECaPSRegionOptions):
                d_steps = round((d_max - d_min) / self._input_number("distance_step"))
            else:
                d_steps = 10
            distances = np.linspace(d_min, d_max, d_steps) * u.pc

            lon3d, lat3d, dist3d = np.meshgrid(
                lon_grid,
                lat_grid,
                distances,
            )
            grid_in_offset_frame = SkyCoord(
                lon=lon3d,
                lat=lat3d,
                distance=dist3d,
                frame=offset_frame,
            )
        else:
            grid_in_offset_frame = SkyCoord(
                lon=lon_grid,
                lat=lat_grid,
                distance=center.distance,
                frame=offset_frame
            )
        
        return grid_in_offset_frame.transform_to('galactic')

    def import_data(self):
        try:
            map_name = self.map_selector.currentText()
            QueryClass, fetch_map, options_cls = MAPS[map_name]

            if (fetch_map is not None) and not os.path.exists(os.path.join(config['data_dir'], map_name.lower())):
                fetch_map()

            coords = self._input_coords(options_cls)
            print(coords)
            dustmap = QueryClass()
            ebv = dustmap(coords)
            print(ebv)
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
