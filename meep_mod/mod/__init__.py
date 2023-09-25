"""
mod:
modified meep methods & costume methods
"""

__all__ = ["geometry", "visualization", "simulation", "io", "env", "ui"]

from .geometry import (vector3_array, vector3_array_xyz)

from .simulation import (step, unit,
                         lists2ndarrays,
                         appendlst, poplst,
                         append_sim_t, append_harminv_modes, append_flux, append_dft, append_volume, append_mode_coeff,
                         output_dft, save_energy, save_flux, save_near, output_times)

from .visualization import (center_plane, plot2D,
                            center_slice, plot_data_eps, plot_data_field)

from .io import (master_do, really_master_do,
                 new_dir,
                 output_h5, read_h5)

from .env import (is_matplotlib_available, with_matplotlib,
                  is_mayavi_available, with_mayavi,
                  is_mpi_available, with_mpi, master_do, really_master_do,
                  get_time, get_rank, get_processor_name)

from .ui import (Model,
                 FdtdStudy, ModeStudy,
                 GeoScript, SrcScript, VolMonScript, DftMonScript)
