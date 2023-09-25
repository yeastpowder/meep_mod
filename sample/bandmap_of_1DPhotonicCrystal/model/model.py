"""
MAIN MODEL
1D photonic crystal
"""

import meep as mp
from meep import mpb
import numpy as np

import meep_mod.mod as mod
try:
    from .modestudy_bandmap_ import ModeStudy_BandMap_
except:
    from modestudy_bandmap_ import ModeStudy_BandMap_  # solve problems for directly running model.py


class MainModel(mod.Model):
    def __init__(self,
                 resolution=20,
                 n_k=15, n_b=3,
                 wg_size=mp.Vector3(mp.inf, 0.7, 0.22), pad_size=mp.Vector3(0, 1, 1),
                 parity=mp.NO_PARITY,
                 a=0.33, ff=0.1,
                 output_dir="./data",
                 _n=None):
        super(MainModel, self).__init__()
        self._n = _n
        # --------- global parameters ----------
        # --- resolution ---
        self.resolution = resolution

        # --- geometry ---
        self.wg_size = wg_size  # wg and sim cell
        self.pad_size = pad_size
        self.cell_size = self.wg_size + self.pad_size * 2
        self.cell_size.x = mp.inf

        self.a = a  # period of unit cell
        self.unit_cell = mp.Vector3(self.a, self.cell_size.y, self.cell_size.z)

        self.ff = ff  # filling factor, calculate hole size
        self.unit_wg = mp.Vector3(self.a, self.wg_size.y, self.wg_size.z)
        unit_wg_area = self.unit_wg.x * self.unit_wg.y  # hole in unit cell
        self.hole_size = mp.Vector3(2 * (np.sqrt(self.ff * unit_wg_area / np.pi)),
                                    2 * (np.sqrt(self.ff * unit_wg_area / np.pi)),
                                    mp.inf)

        # --- parity ---
        self.parity = parity

        # --- band map  ---
        self.n_k = n_k  # k points
        self.k_hpi_mp = mp.Vector3(x=0.5 / self.a)
        self.k_points_mp = mp.interpolate(self.n_k, [mp.Vector3(0), self.k_hpi_mp])
        self.k_hpi_mpb = mp.Vector3(x=0.5)
        self.k_points_mpb = mp.interpolate(self.n_k, [mp.Vector3(0), self.k_hpi_mpb])
        self.n_b = n_b  # n bands

        # --- output homedir ---
        self.output_dir = output_dir

        # ---------- material ----------
        self.Air = mp.Medium(epsilon=1)
        self.Si = mp.Medium(epsilon=12.087)

        #  ---------- geometry ----------
        self.wg = [mp.Block(size=self.wg_size, material=self.Si)]
        self.etch = [mp.Ellipsoid(size=self.hole_size, center=mp.Vector3())]

        # --------- simulation ----------
        self.ms = mpb.ModeSolver(resolution=self.resolution,
                                 geometry_lattice=mp.Lattice(size=self.unit_cell),
                                 num_bands=self.n_b, k_points=self.k_points_mpb)

        # ---------- study ----------
        self.Mode_bandmap_ = ModeStudy_BandMap_(self, parity=self.parity, _n=self._n, output_dir=self.output_dir)


# %% __main__ visualize
if __name__ == '__main__':
    model = MainModel()

    study = model.Mode_bandmap_
    ms = model.ms
    ms.geometry = model.wg + model.etch
    ms.init_params(model.parity, True)
    study.plot2D(ms)

