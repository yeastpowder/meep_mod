"""
MAIN MODEL
1D photonic crystal
"""

import meep as mp
from meep import mpb
import numpy as np

import meep_mod.mod as mod
try:
    from .fdtdstudy_flux_ import FDTDStudy_Flux_
    from .geo_1DBraggMirror_ import Geo_1DBraggMirror_
except:
    from fdtdstudy_flux_ import FDTDStudy_Flux_
    from geo_1DBraggMirror_ import Geo_1DBraggMirror_  # solve problems for directly running model.py
from meep_mod.gallery import StdMon_Flux_


class MainModel(mod.Model):
    def __init__(self,
                 resolution=20,
                 f_cen=0.66, f_width=0.1, f_num=3,
                 wg_size=mp.Vector3(mp.inf, 0.7, 0.22), pad_size=mp.Vector3(2, 2, 2), pml_width=0.5,
                 src2mirror=mp.Vector3(4),
                 symmetries=[],
                 a=0.33, ff=0.1, n_holes=15,
                 dt=5,
                 output_dir="./data",
                 _n=None):
        super(MainModel, self).__init__()
        self._n = _n
        # --------- global parameters ----------
        # --- resolution ---
        self.resolution = resolution

        # --- geometry ---
        self.wg_size = wg_size  # wg
        self.pad_size = pad_size

        self.a = a  # period of unit cell

        self.ff = ff  # filling factor, calculate hole size
        self.unit_wg = mp.Vector3(self.a, self.wg_size.y, self.wg_size.z)
        unit_wg_area = self.unit_wg.x * self.unit_wg.y  # hole in unit cell
        self.hole_size = mp.Vector3(2 * (np.sqrt(self.ff * unit_wg_area / np.pi)),
                                    2 * (np.sqrt(self.ff * unit_wg_area / np.pi)),
                                    mp.inf)

        self.n_holes = n_holes

        self.src2mirror = src2mirror

        self.cell_size = mp.Vector3(self.src2mirror.x+self.a*(self.n_holes-1),
                                    wg_size.y,
                                    wg_size.z) + self.pad_size * 2          # sim cell

        self.pml_width = pml_width

        self.src_pt = mp.Vector3(-self.cell_size.x/2+self.pad_size.x)
        self.mirror_start_pt = self.src_pt + self.src2mirror

        self.src_size = self.wg_size + mp.Vector3(0,0.5,0.5)
        self.src_size.x = 0
        self.mon_size = self.src_size

        # --- parity ---
        self.symmetries = symmetries

        # --- frequency  ---
        self.f_cen = f_cen
        self.f_width = f_width
        self.f_num = f_num

        # --- time step ---
        self.dt = dt

        # --- output homedir ---
        self.output_dir = output_dir

        # ---------- material ----------
        self.Air = mp.Medium(epsilon=1)
        self.Si = mp.Medium(epsilon=12.087)

        #  ---------- geometry ----------
        self.geometry_ = Geo_1DBraggMirror_(self,
                                            wg_size=self.wg_size, wg_center=mp.Vector3(),
                                            start_pt=mp.Vector3(), a=self.a, ff=self.ff, n_holes=self.n_holes,
                                            material=self.Si)

        # ---------- boundary ----------
        self.pmls = [mp.PML(self.pml_width)]

        # ---------- src ----------
        self.eigen_src = [mp.EigenModeSource(mp.GaussianSource(frequency=self.f_cen, fwidth=1.5 * self.f_width),
                                             volume=mp.Volume(center=self.src_pt, size=self.src_size))]

        # ---------- monitor ----------
        self.fluxs_ = [StdMon_Flux_(self, center=self.src_pt+self.src2mirror/2, size=self.mon_size,
                                    f_cen=self.f_cen,
                                    f_width=self.f_width,
                                    n_freq=self.f_num)]  # flux after src
        self.fluxs_ += [StdMon_Flux_(self, center=pt, size=self.mon_size,
                                    f_cen=self.f_cen,
                                    f_width=self.f_width,
                                    n_freq=self.f_num) for pt in self.geometry_.flux_pts]  # fluxes in Phc

        # --------- simulation ----------
        #   -♜ sim obj
        self.sim = mp.Simulation(cell_size=self.cell_size, resolution=self.resolution,
                                 sources=self.eigen_src,
                                 boundary_layers=self.pmls,
                                 symmetries=self.symmetries,
                                 geometry=self.geometry_.create_geometry())

        # ---------- study ----------
        self.FDTD_flux_ = FDTDStudy_Flux_(self, fluxes_=self.fluxs_, dt=dt, _n=self._n, output_dir=self.output_dir)


# %% __main__ visualize
if __name__ == '__main__':
    model = MainModel(symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=1)])

    study = model.FDTD_flux_
    sim = model.sim
    study.plot2D(sim)

