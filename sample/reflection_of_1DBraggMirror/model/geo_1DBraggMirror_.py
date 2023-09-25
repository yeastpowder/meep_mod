"""
geometry_
1D BraggMirror
"""

import meep as mp
import numpy as np

import meep_mod.mod as mod


class Geo_1DBraggMirror_(mod.GeoScript):
    def __init__(self, model,
                 wg_size, wg_center,
                 start_pt, a, ff, n_holes,
                 material):
        super(Geo_1DBraggMirror_, self).__init__(model)
        # ---------- locals ----------
        # ---- geo parameters ---
        self.wg_size = wg_size  # waveguide
        self.wg_center = wg_center

        self.start_pt = start_pt  # bragg mirror
        self.a = a
        self.ff = ff
        self.n_holes = n_holes
        self._hole_pts = mod.vector3_array_xyz(self.start_pt.x + np.arange(0, self.n_holes)*self.a,
                                               self.start_pt.y,
                                               self.start_pt.z)
        self._hole_size = mp.Vector3(2 * (np.sqrt(self.ff * self.a * self.wg_size.y / np.pi)),
                                     2 * (np.sqrt(self.ff * self.a * self.wg_size.y / np.pi)),
                                     mp.inf)

        # --- material ---
        self.material = material

        # --- flux points ---
        self.flux_pts = [hole_pt + mp.Vector3(a/2) for hole_pt in self._hole_pts]

    def create_geometry(self):
        wg = [mp.Block(size=self.wg_size, center=self.wg_center, material=self.material)]
        etch = list(map(lambda pt: mp.Ellipsoid(size=self._hole_size, center=pt), self._hole_pts))
        return wg+etch

