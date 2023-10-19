"""test_project/model/model.py"""

import numpy as np

import meep as mp
import meep_mod.mod as mod
try:
    from .fdtdstudy_flux_ import FDTDStudy_Flux_
    from .geo_wg_ import Geo_wg_
except:
    from fdtdstudy_flux_ import FDTDStudy_Flux_
    from geo_wg_ import Geo_wg_  # solve problems for directly running model.py
from meep_mod.gallery import StdMon_Flux_


class MainModel(mod.Model):
    def __init__(self,
                 resolution=20,
                 f_cen=0.66, f_width=0.1, f_num=3,
                 wg_l=5, wg_h=0.22, wg_w0=5, wg_w1=0.7,
                 pad_size=mp.Vector3(0, 2, 2), pml_width=0.5,
                 src2pml=mp.Vector3(0.5), mon2pml=mp.Vector3(1),
                 symmetries=[],
                 dt=5,
                 output_dir="./data",
                 _n=None):
        super(MainModel, self).__init__()
        self._n = _n
        # --------- global parameters ----------
        # --- resolution ---
        self.resolution = resolution

        # --- geometry ---
        self.wg_l = wg_l  # wg
        self.wg_h = wg_h
        self.wg_w0, self.wg_w1 = wg_w0, wg_w1
        self.pad_size = pad_size

        self.src2pml = src2pml
        self.mon2pml = mon2pml

        self.cell_size = mp.Vector3(self.wg_l,
                                    max(self.wg_w0, self.wg_w1),
                                    self.wg_h) + self.pad_size * 2          # sim cell

        self.pml_width = pml_width

        self.src_pt = mp.Vector3(-self.cell_size.x/2+self.pad_size.x+self.src2pml.x)
        self.mon_pt = mp.Vector3(-self.cell_size.x/2 + self.pad_size.x + self.mon2pml.x)

        self.src_size = mp.Vector3(0, self.wg_w0, self.wg_h) + mp.Vector3(0,0.5,0.5)
        self.mon_size0 = self.src_size
        self.mon_size1 = mp.Vector3(0, self.wg_w1, self.wg_h) + mp.Vector3(0,1,0.5)

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
        self.geometry_ = Geo_wg_(self,
                                 wg_l=self.wg_l, wg_h=self.wg_h, wg_w0=self.wg_w0, wg_w1=self.wg_w1, wg_center=mp.Vector3(),
                                 wg_material=self.Si)

        # ---------- boundary ----------
        self.pmls = [mp.PML(self.pml_width)]

       # ---------- src ----------
        self.eigen_src = [mp.EigenModeSource(mp.GaussianSource(frequency=self.f_cen, fwidth=1.5 * self.f_width),
                                             volume=mp.Volume(center=self.src_pt, size=self.src_size))]

        # ---------- monitor ----------
        self.fluxs_ = [StdMon_Flux_(self, center=self.mon_pt, size=self.mon_size0,
                                    f_cen=self.f_cen,
                                    f_width=self.f_width,
                                    n_freq=self.f_num)]  # flux i
        self.fluxs_ += [StdMon_Flux_(self, center=-1*self.mon_pt, size=self.mon_size1,
                                     f_cen=self.f_cen,
                                     f_width=self.f_width,
                                     n_freq=self.f_num)]  # flux o

        # --------- simulation ----------
        #   --- sim obj ---
        self.sim = mp.Simulation(cell_size=self.cell_size, resolution=self.resolution,
                                 sources=self.eigen_src,
                                 boundary_layers=self.pmls,
                                 symmetries=self.symmetries,
                                 geometry=self.geometry_.create_geometry())

        # ---------- study ----------
        self.FDTD_flux_ = FDTDStudy_Flux_(self, fluxes_=self.fluxs_, dt=dt, _n=self._n, output_dir=self.output_dir)


if __name__ == '__main__':
    model = MainModel(symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=1)])

    study = model.FDTD_flux_
    sim = model.sim
    study.plot2D(sim)
