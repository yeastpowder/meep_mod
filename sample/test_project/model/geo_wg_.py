"""test_project/model/geo_wg_.py"""

import meep as mp
import meep_mod.mod as mod


class Geo_wg_(mod.GeoScript):
    def __init__(self, model,
                 wg_l, wg_h, wg_w0, wg_w1, wg_center, wg_material,
                 sub_size=None, sub_material=None):
        super(Geo_wg_, self).__init__(model)
        # ---------- locals ----------
        # ---- geo.jpg parameters ---
        self.wg_l = wg_l  # waveguide
        self.wg_h = wg_h
        self.wg_w0 = wg_w0
        self.wg_w1 = wg_w1
        self.wg_center = wg_center

        self._v1 = wg_center + mp.Vector3(-wg_l/2,-wg_w0/2,-wg_h/2)
        self._v2 = wg_center + mp.Vector3(-wg_l/2,+wg_w0/2,-wg_h/2)
        self._v3 = wg_center + mp.Vector3(wg_l/2,+wg_w1/2,-wg_h/2)
        self._v4 = wg_center + mp.Vector3(wg_l/2,-wg_w1/2,-wg_h/2)

        self.sub_size = sub_size  # substrate
        if self.sub_size is not None:
            self.sub_center = mp.Vector3(wg_center.x, wg_center.y, wg_center.z-(wg_h+sub_size.z)/2)

        # --- material ---
        self.wg_material = wg_material
        self.sub_material = sub_material

    def create_geometry(self):
        wg = [mp.Prism([self._v1, self._v2, self._v3, self._v4],
                       height=self.wg_h, axis=mp.Vector3(0,0,1),
                       material=self.wg_material)]
        if self.sub_size is not None and self.sub_material is not None:
            sub = [mp.Block(size=self.sub_size, center=self.sub_center, material=self.sub_material)]
        else:
            sub = []
        return wg+sub


if __name__ == '__main__' and mod.is_matplotlib_available():  # test
    import matplotlib.pyplot as plt

    wg_ = Geo_wg_(None,
                  wg_l=5, wg_h=0.22, wg_w0=1.5, wg_w1=0.7, wg_center=mp.Vector3(), wg_material=mp.Medium(epsilon=12),
                  sub_size=mp.Vector3(5,3,2), sub_material=mp.Medium(epsilon=3))

    sim = mp.Simulation(resolution=10, cell_size=mp.Vector3(5,2,3))
    sim.geometry = wg_.create_geometry()

    fig, axs = plt.subplots(3,1)
    mod.plot2D(sim, fig, axs[0], output_plane=mod.center_plane(sim.cell_size, ax='z'))
    mod.plot2D(sim, fig, axs[1], output_plane=mod.center_plane(sim.cell_size, ax='x'))
    mod.plot2D(sim, fig, axs[2], output_plane=mod.center_plane(sim.cell_size, ax='y'))
    plt.show()