"""
STANDARD MONITOR_
mode decomposition
"""

import meep as mp
import meep_mod.mod as mod
from meep_mod.gallery.monitor_ import StdMon_Flux_


class StdMon_Mode_(StdMon_Flux_):
    def __init__(self, model, **kwargs):
        super(StdMon_Mode_, self).__init__(model, **kwargs)
        # -locals
        # -passive locals declaration
        pass

    def add_monitors(self, sim):
        # -passive locals
        pass
        # -create
        if self.flux_regs is None:
            self.flux_regs = [mp.FluxRegion(
                center=self.center, size=self.size, direction=self.direction, weight=self.weight, volume=None)]
        if self.freq:
            mon = mp.Simulation.add_mode_monitor(sim, self.freq, self.flux_regs)
        else:
            mon = mp.Simulation.add_mode_monitor(sim, self.f_cen, self.f_width, self.n_freq, *self.flux_regs)
        return [mon]


# %% __main__ test
if __name__ == '__main__' and mod.is_matplotlib_available():
    import matplotlib.pyplot as plt

    test_ = StdMon_Mode_(None, center=mp.Vector3(), size=mp.Vector3(0, 0.5, 0.5),
                         f_cen=0.66, f_width=0.1)
    # test_ = StdMon_Flux_(None, center=mp.Vector3(), size=mp.Vector3(0, 0.5, 0.5),
    #                     freq=[0.59, 0.6, 0.61])

    sim = mp.Simulation(resolution=10, cell_size=mp.Vector3(1, 1, 1))
    m = test_.add_monitors(sim)

    fig, ax = plt.subplots(1, 1)
    mod.plot2D(sim, fig, ax, output_plane=mod.center_plane(cell=sim.cell_size, ax='z'))
    plt.show()
