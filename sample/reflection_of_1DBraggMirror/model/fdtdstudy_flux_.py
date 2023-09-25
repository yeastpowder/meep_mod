"""
STUDY_
FDTD
flux
"""

import meep as mp
import numpy as np

import meep_mod.mod as mod
from meep_mod.gallery.study_ import StdStudy_FDTD_


class FDTDStudy_Flux_(StdStudy_FDTD_):
    def __init__(self, model, fluxes_, dt=mp.inf, **kwargs):
        super(FDTDStudy_Flux_, self).__init__(model, dft_monitors_=fluxes_, **kwargs)
        # ---------- locals ----------
        # --- monitors ---
        self.dt = dt
        # --- data ---
        self.fluxes_history = None
        self.fluxes_freq = None
        self.t = None

    def _create_step_funs(self):
        """ run after ._create_dfts() """
        # create empty data spaces
        self.fluxes_history = []
        self.fluxes_freq = []
        self.t = []
        # create step funcs
        append_flux = []
        append_t = []
        convert_nd_data = []
        for i, flux in enumerate(self._dfts):
            self.fluxes_history.append([])
            self.fluxes_freq.append(mp.get_flux_freqs(flux))

            append_flux += [mp.at_every(self.dt, mod.append_flux(flux, self.fluxes_history[i]))]
            append_flux += [mp.at_end(mod.append_flux(flux, self.fluxes_history[i]))]

        append_t += [mp.at_every(self.dt, mod.append_sim_t(self.t))]
        append_t += [mp.at_end(mod.append_sim_t(self.t))]

        convert_nd_data += [mp.at_end(mod.lists2ndarrays(self, ["fluxes_history", "t", "fluxes_freq"]))]

        self._step_funcs = append_flux + append_t + convert_nd_data
        return self._step_funcs

    def run_study(self, sim: mp.Simulation, output_field=False, **kwargs):
        self._create_dir()
        self._create_dfts(sim)
        self._create_vols()
        self._create_step_funs()

        if output_field:
            sim.filename_prefix = self._runtime_dir + '/' + 'field'  # overwrite output dir of sim
            self._step_funcs += [mp.at_every(self.dt, mp.output_efield)]

        mp.Simulation.run(sim,
                          *self._step_funcs,
                          **kwargs)

