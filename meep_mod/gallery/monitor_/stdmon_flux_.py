"""
STANDARD MONITOR_
flux
"""

import meep as mp
import meep_mod.mod as mod


class StdMon_Flux_(mod.DftMonScript):
    def __init__(self, model,
                 center=None, size=None, weight=1.0, direction=mp.AUTOMATIC,
                 flux_regs=None,
                 f_cen=None, f_width=None, n_freq=10, freq=None):
        """
        flux 监视器，测量 （center, size）确定的plane内 能流通量 的频谱展开
        :param model: mod.Model: parent model
        :param center: mp.Vector3: plane center
        :param size: mp.Vector3: plane size
        :param weight:
        :param direction:
        :param flux_regs: use [mp.FluxRegion] instead to init
        :param f_cen: arg for multi freq
        :param f_width: arg for multi freq
        :param n_freq: arg for multi freq
        :param freq: arg for single freq
        """
        super(StdMon_Flux_, self).__init__(model)
        # -locals
        self.center = center
        self.size = size
        self.weight = weight
        self.direction = direction
        self.flux_regs = flux_regs
        self.f_cen = f_cen
        self.f_width = f_width
        self.n_freq = n_freq
        self.freq = freq
        # -passive locals declaration
        pass

    def add_monitors(self, sim):
        """
        向 simulation 加入 monitor，原函数如下
        mp.Simulation.add_flux(fcen, df, nfreq, freq, FluxRegions, decimation_factor=1)
        mp.FluxRegion(self, center=None, size=Vector3(), direction=mp.AUTOMATIC, weight=1.0, volume=None)
        """
        # -passive locals
        pass
        # -create
        if self.flux_regs is None:
            self.flux_regs = [mp.FluxRegion(
                center=self.center, size=self.size, direction=self.direction, weight=self.weight, volume=None)]
        if self.freq:
            mon = mp.Simulation.add_flux(sim, self.freq, self.flux_regs)
        else:
            mon = mp.Simulation.add_flux(sim, self.f_cen, self.f_width, self.n_freq, *self.flux_regs)
        return [mon]


# %% __main__ test
if __name__ == '__main__' and mod.is_matplotlib_available():
    import matplotlib.pyplot as plt

    test_ = StdMon_Flux_(None, center=mp.Vector3(), size=mp.Vector3(0, 0.5, 0.5),
                         f_cen=0.66, f_width=0.1)
    # test_ = StdMon_Flux_(None, center=mp.Vector3(), size=mp.Vector3(0, 0.5, 0.5),
    #                     freq=[0.59, 0.6, 0.61])

    sim = mp.Simulation(resolution=10, cell_size=mp.Vector3(1, 1, 1))
    m = test_.add_monitors(sim)

    fig, ax = plt.subplots(1, 1)
    mod.plot2D(sim, fig, ax, output_plane=mod.center_plane(cell=sim.cell_size, ax='z'))
    plt.show()
