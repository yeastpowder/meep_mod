"""mod of meep.visualization"""

import meep as mp
import numpy as np
from meep import Vector3, Volume
from meep.visualization import get_2D_dimensions, filter_dict
from meep.visualization import plot_volume, plot_boundaries, plot_monitors, plot_sources
from meep.visualization import default_field_parameters


# %% matplotlib ver check
def matplotlib_version_check():
    import matplotlib
    ver = matplotlib.__version__
    print("matplotlib ver =", ver, "tested on ver 3.4.2")

    return ver


# %% Modify default para

default_eps_parameters = {
    'interpolation': 'spline36',
    'cmap': 'binary',
    'alpha': 1.0
}

default_volume_parameters = {
        'color':'y',
        'edgecolor':'y',
        'facecolor':'none',
        'hatch':'/',
        'linewidth':2,
        'alpha': 0.6
    }


# %% plot2D
def center_plane(cell: mp.Vector3(), ax='z', offset=0):
    """ get a mp.Volume-type center_plane of cell, used in plot(output_plane) """
    size = mp.Vector3(*cell)    # size is a new obj, do not use size = cell
    center = mp.Vector3()

    if ax == 'x':
        size.x = 0
        center.x = offset
    elif ax == 'y':
        size.y = 0
        center.y = offset
    elif ax == 'z':
        size.z = 0
        center.z = offset
    else:
        raise ValueError("Wrong ax parameter")

    return mp.Volume(center=center, size=size)


def center_slice(data: np.array([[[]]]), ax='z'):
    """ get a center_slice: n*2 array from data, used in plot(data)"""
    shape = data.shape

    if ax == 'x':
        center = int(np.floor(shape[0] / 2))
        return data[center, :, :]
    elif ax == 'y':
        center = int(np.floor(shape[1] / 2))
        return data[:, center, :]
    elif ax == 'z':
        center = int(np.floor(shape[2] / 2))
        return data[:, :, center]
    else:
        raise ValueError("Wrong ax parameter")


"""Modified build-in plot functional"""


def plot_eps(sim, fig, ax, output_plane=None, eps_parameters=None):
    if sim.structure is None:
        sim.init_sim()

    # consolidate plotting parameters
    eps_parameters = default_eps_parameters if eps_parameters is None else dict(default_eps_parameters,
                                                                                **eps_parameters)

    # Get domain measurements
    sim_center, sim_size = get_2D_dimensions(sim, output_plane)

    xmin = sim_center.x - sim_size.x / 2
    xmax = sim_center.x + sim_size.x / 2
    ymin = sim_center.y - sim_size.y / 2
    ymax = sim_center.y + sim_size.y / 2
    zmin = sim_center.z - sim_size.z / 2
    zmax = sim_center.z + sim_size.z / 2

    center = Vector3(sim_center.x, sim_center.y, sim_center.z)
    cell_size = Vector3(sim_size.x, sim_size.y, sim_size.z)

    if sim_size.x == 0:
        # Plot y on x axis, z on y axis (YZ plane)
        extent = [ymin, ymax, zmin, zmax]
        xlabel = 'Y'
        ylabel = 'Z'
    elif sim_size.y == 0:
        # Plot x on x axis, z on y axis (XZ plane)
        extent = [xmin, xmax, zmin, zmax]
        xlabel = 'X'
        ylabel = 'Z'
    elif sim_size.z == 0:
        # Plot x on x axis, y on y axis (XY plane)
        extent = [xmin, xmax, ymin, ymax]
        xlabel = 'X'
        ylabel = 'Y'
    else:
        raise ValueError("A 2D plane has not been specified...")

    eps_data = np.rot90(np.real(sim.get_array(center=center, size=cell_size, component=mp.Dielectric)))
    if mp.am_master():
        im = ax.imshow(eps_data, extent=extent, **eps_parameters)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax)

    return ax


def plot_fields(sim, fig=None, ax=None, fields=None, output_plane=None, field_parameters=None):
    if not sim._is_initialized:
        sim.init_sim()

    if fields is None:
        return ax

    field_parameters = default_field_parameters if field_parameters is None else dict(default_field_parameters,
                                                                                      **field_parameters)

    # user specifies a field component
    if fields in [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]:
        # Get domain measurements
        sim_center, sim_size = get_2D_dimensions(sim, output_plane)

        xmin = sim_center.x - sim_size.x / 2
        xmax = sim_center.x + sim_size.x / 2
        ymin = sim_center.y - sim_size.y / 2
        ymax = sim_center.y + sim_size.y / 2
        zmin = sim_center.z - sim_size.z / 2
        zmax = sim_center.z + sim_size.z / 2

        center = Vector3(sim_center.x, sim_center.y, sim_center.z)
        cell_size = Vector3(sim_size.x, sim_size.y, sim_size.z)

        if sim_size.x == 0:
            # Plot y on x axis, z on y axis (YZ plane)
            extent = [ymin, ymax, zmin, zmax]
            xlabel = 'Y'
            ylabel = 'Z'
        elif sim_size.y == 0:
            # Plot x on x axis, z on y axis (XZ plane)
            extent = [xmin, xmax, zmin, zmax]
            xlabel = 'X'
            ylabel = 'Z'
        elif sim_size.z == 0:
            # Plot x on x axis, y on y axis (XY plane)
            extent = [xmin, xmax, ymin, ymax]
            xlabel = 'X'
            ylabel = 'Y'
        fields = sim.get_array(center=center, size=cell_size, component=fields)
    else:
        raise ValueError('Please specify a valid field component (mp.Ex, mp.Ey, ...')

    fields = field_parameters['post_process'](fields)

    # Either plot the field, or return the array
    if ax:
        if mp.am_master():
            im = ax.imshow(np.rot90(fields), extent=extent, **filter_dict(field_parameters, ax.imshow))
            fig.colorbar(im, ax=ax)
        return fig, ax
    else:
        return np.rot90(fields)


def plot2D(sim, fig, ax=None, output_plane=None, fields=None, labels=False,
           eps_parameters=None, boundary_parameters=None,
           source_parameters=None, monitor_parameters=None,
           field_parameters=None,
           field_on=True, eps_on=True, source_on=True, monitor_on=True, boundary_on=True,
           volumes=None
           ):
    # Initialize the simulation
    if sim.structure is None:
        sim.init_sim()
    # Ensure a figure axis exists
    if ax is None and mp.am_master():
        from matplotlib import pyplot as plt
        ax = plt.gca()

    # validate the output plane to ensure proper 2D coordinates
    from meep.simulation import Volume
    sim_center, sim_size = get_2D_dimensions(sim, output_plane)
    output_plane = Volume(center=sim_center, size=sim_size)

    # Plot geometry
    if eps_on:
        ax = plot_eps(sim, fig, ax, output_plane=output_plane, eps_parameters=eps_parameters)

    # Plot boundaries
    if boundary_on:
        plot_boundaries(sim, ax, output_plane=output_plane, boundary_parameters=boundary_parameters)

    # Plot sources
    if source_on:
        plot_sources(sim, ax, output_plane=output_plane, labels=labels, source_parameters=source_parameters)

    # Plot monitors
    if monitor_on:
        plot_monitors(sim, ax, output_plane=output_plane, labels=labels, monitor_parameters=monitor_parameters)

    # Plot fields
    if field_on:
        plot_fields(sim, fig, ax, fields, output_plane=output_plane, field_parameters=field_parameters)

    # Plot extra volumes
    if volumes:
        for volume in volumes:
            plot_volume(sim, ax, volume, output_plane=output_plane, plotting_parameters=default_volume_parameters)

    return ax


"""New plot functional"""


def plot_data_eps(fig, ax, eps_data_slice, size=None, alpha=None):
    if size:
        extent = [-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2]
    else:
        extent = None

    if alpha is None:
        alpha = 1.0

    # plt rot90
    im = ax.imshow(np.rot90(eps_data_slice), interpolation='spline36', cmap='binary',
                   extent=extent, alpha=alpha)
    fig.colorbar(im, ax=ax)

    return im


def plot_data_field(fig, ax, field_data_slice, size=None, alpha=None):
    if size:
        extent = [-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2]
    else:
        extent = None

    if alpha is None:
        alpha = 0.5

    im = ax.imshow(np.rot90(field_data_slice), interpolation='spline36', cmap='RdBu',
                   alpha=alpha, extent=extent)
    fig.colorbar(im, ax=ax)

    return im


