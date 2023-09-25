"""
STANDARD MONITOR_
dft near2far
"""

import meep as mp
import meep_mod.mod as mod


class StdMon_Near2Far_(mod.DftMonScript):
    def __init__(self, model,
                 center=None, size=None, weight=1.0, near_regs=None,
                 f_cen=None, f_width=None, n_freq=10, freq=None):
        """
        DFT 监视器，测量 （center, size）确定的box边界场 由此计算 远场辐射 的频谱展开
        :param model: mod.Model: parent model
        :param center: mp.Vector3: box center
        :param size: mp.Vector3: box size
        :param weight:
        :param near_regs: use [mp.Near2FarRegion] instead to init
        :param f_cen: arg for multi freq
        :param f_width: arg for multi freq
        :param n_freq: arg for multi freq
        :param freq: arg for single freq
        """
        super(StdMon_Near2Far_, self).__init__(model)
        # -locals
        self.center = center
        self.size = size
        self.weight = weight
        self.near_regs = near_regs
        self.f_cen = f_cen
        self.f_width = f_width
        self.n_freq = n_freq
        self.freq = freq
        # -passive locals declaration
        pass

    def add_monitors(self, sim):
        """
        向 simulation 加入 monitor，原函数如下
        mp.Simulation.add_near2far(fcen, df, nfreq, freq, Near2FarRegions, nperiods=1, decimation_factor=1)`  ##sig
        mp.FluxRegions(self, center=None, size=Vector3(), direction=mp.AUTOMATIC, weight=1.0, volume=None):
        """
        # -passive locals
        pass
        # -create
        if self.near_regs is None:
            self.near_regs = [mp.Near2FarRegion(
                center=self.center, size=self.size, weight=self.weight, volume=None)]
        if self.freq:
            mon = mp.Simulation.add_near2far(sim, self.freq, self.near_regs)
        else:
            mon = mp.Simulation.add_near2far(sim, self.f_cen, self.f_width, self.n_freq, *self.near_regs)
        return [mon]


# %% __main__ test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    f = 0.66
    df = 0.1
    test_ = StdMon_Near2Far_(None, center=mp.Vector3(y=1), size=mp.Vector3(9, 0, 2),
                             f_cen=f, f_width=df)
    # a phc structure
    etch = list(map(lambda size, center: mp.Ellipsoid(size=size, center=center),
                         mod.vector3_array_xyz([0.5]*10, [0.5]*10, mp.inf),
                         mod.vector3_array_xyz(x=list((i+0.5)*0.8 for i in range(-10, 10)))))
    wg = [mp.Block(size=mp.Vector3(mp.inf, 0.8, 0.22), material=mp.Medium(epsilon=12))]
    sim = mp.Simulation(resolution=10, cell_size=mp.Vector3(10, 4, 3),
                        boundary_layers=[mp.PML(0.5)],
                        geometry=wg+etch,
                        sources=[mp.Source(mp.GaussianSource(frequency=f, fwidth=3 * df), component=mp.Ey,
                                           center=mp.Vector3(), size=mp.Vector3())]
                        )
    m = test_.add_monitors(sim)

    fig, ax = plt.subplots(1, 1)
    mod.plot2D(sim, fig, ax, output_plane=mod.center_plane(cell=sim.cell_size, ax='z'))
    plt.show()
    fig, ax = plt.subplots(1, 1)
    mod.plot2D(sim, fig, ax, output_plane=mod.center_plane(cell=sim.cell_size, ax='x'))
    plt.show()
    # run
    sim.run(until_after_sources=1000)
    fig, ax = plt.subplots(1, 1)
    mod.plot2D(sim, fig, ax, output_plane=mod.center_plane(cell=sim.cell_size, ax='z'), fields=mp.Ey)
    plt.show()

    # far field vs frq
    far_field = sim.get_farfields(m[0], resolution=2, center=mp.Vector3(y=10), size=mp.Vector3(20,5,1))

    fig = plt.figure(figsize=(3, 7), dpi=100)
    axs = fig.subplots(10, 1)
    for i, ax in enumerate(axs):
        mod.plot_data_field(fig, ax, np.real(mod.center_slice(far_field['Ey'][:,:,:,i],ax='z')))
    plt.show()
    # far field polar plot
    radius = 5
    n_theta = 50
    thetas = []
    fluxes = []
    for i in range(n_theta):
        theta = np.pi*i/n_theta
        thetas.append(theta)
        pt = mp.Vector3(x=radius*np.cos(theta), y=radius*np.sin(theta))
        far_field = np.array(sim.get_farfield(m[0], pt))
        far_field = far_field.reshape((int(len(far_field)/6), 6))
        E = far_field[:, :3]
        H = far_field[:, 3:]
        flux = np.real(np.cross(np.conj(E), H))**2
        flux = np.sqrt(np.sum(flux, axis=-1))
        fluxes.append(flux)
    thetas = np.array(thetas)
    fluxes = np.array(fluxes)

    thetas = np.hstack([thetas, -1*thetas])
    fluxes = np.vstack([fluxes, fluxes])

    plt.polar(thetas, fluxes[:, 2])
    plt.show()

