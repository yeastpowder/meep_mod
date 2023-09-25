"""
STANDARD MONITOR_
dft field
"""

import meep as mp
import meep_mod.mod as mod


class StdMon_Dft_(mod.DftMonScript):
    def __init__(self, model,
                 center, size,
                 cs=[mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
                 f_cen=None, f_width=None, n_freq=10, freq=None):
        """
        DFT 监视器，测量 （center, size）确定的box内 场分量cs 的频谱展开
        :param model: mod.Model: parent model
        :param center: mp.Vector3: box center
        :param size: mp.Vector3: box size
        :param cs: e.g. [mp.Ex, mp.Ey, ...]
        :param f_cen: arg for multi freq
        :param f_width: arg for multi freq
        :param n_freq: arg for multi freq
        :param freq: arg for single freq
        """
        super(StdMon_Dft_, self).__init__(model)
        # -locals
        self.center = center
        self.size = size
        self.cs = cs
        self.f_cen = f_cen
        self.f_width = f_width
        self.n_freq = n_freq
        self.freq = freq
        # -passive locals declaration
        pass

    def add_monitors(self, sim):
        """
        向 simulation 加入 monitor，原函数如下
        mp.Simulation.add_dft_fields(self, cs, fcen, df, nfreq, freq, where=None, center=None, size=None,
                                     yee_grid=False, decimation_factor=1)
        """
        # -passive locals
        pass
        # -create
        if self.freq:
            mon = mp.Simulation.add_dft_fields(sim, self.cs, self.freq, center=self.center, size=self.size)
        else:
            mon = mp.Simulation.add_dft_fields(sim, self.cs, self.f_cen, self.f_width, self.n_freq,
                                               center=self.center, size=self.size)
        return [mon]


# %% __main__ test
if __name__ == '__main__' and mod.is_matplotlib_available():
    import matplotlib.pyplot as plt

    test_ = StdMon_Dft_(None, center=mp.Vector3(), size=mp.Vector3(1, 1, 1),
                        f_cen=0.66, f_width=0.1)
    # test_ = StdMon_Dft_(None, center=mp.Vector3(), size=mp.Vector3(0, 0.5, 0.5),
    #                  freq=[0.59, 0.6, 0.61])

    sim = mp.Simulation(resolution=10, cell_size=mp.Vector3(1, 1, 1))
    m = test_.add_monitors(sim)

    fig, ax = plt.subplots(1, 1)
    mod.plot2D(sim, fig, ax, output_plane=mod.center_plane(cell=sim.cell_size, ax='z'))
    plt.show()

