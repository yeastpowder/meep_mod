"""
STANDARD MONITOR_
volume field
"""

import meep as mp
import meep_mod.mod as mod


class StdMon_Vol_(mod.VolMonScript):
    def __init__(self, model,
                 center, size,
                 cs=[mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]):
        """
        DFT 监视器，测量 （center, size）确定的box内 场分量cs
        :param model: mod.Model: parent model
        :param center: mp.Vector3: box center
        :param size: mp.Vector3: box size
        :param cs: e.g. [mp.Ex, mp.Ey, ...]
        """
        super(StdMon_Vol_, self).__init__(model)
        # -locals
        self.center = center
        self.size = size
        self.cs = cs
        # -passive locals declaration
        pass

    def create_volumes(self):
        return [mp.Volume(center=self.center, size=self.size)]

    def get_arrays_from_sim(self, sim):
        """
        mp.Simulation.get_array(self, component=None, vol=None, center=None, size=None,
                                cmplx=None, arr=None, frequency=0, snap=False):
        """
        arrays = []
        for component in self.cs:
            arrays.append(mp.Simulation.get_array(sim, component=component, center=self.center, size=self.size))
        return arrays


# %% __main__ test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_ = StdMon_Vol_(None, center=mp.Vector3(), size=mp.Vector3(1, 1, 1))
    # test_ = StdMon_Dft_(None, center=mp.Vector3(), size=mp.Vector3(0, 0.5, 0.5),
    #                  freq=[0.59, 0.6, 0.61])

    sim = mp.Simulation(resolution=10, cell_size=mp.Vector3(1, 1, 1),
                        sources=[mp.Source(mp.GaussianSource(frequency=0.6, fwidth=0.1), component=mp.Ey,
                                           center=mp.Vector3(), size=mp.Vector3())])

    fig, ax = plt.subplots(1, 1)
    mod.plot2D(sim, fig, ax, volumes=test_.create_volumes(), output_plane=mod.center_plane(cell=sim.cell_size, ax='z'))
    plt.show()

    sim.run(until_after_sources=100)
    fields = test_.get_arrays_from_sim(sim)
    fig, ax = plt.subplots(1, 1)
    mod.plot_data_field(fig, ax, mod.center_slice(fields[1], ax='z'))
    plt.show()

