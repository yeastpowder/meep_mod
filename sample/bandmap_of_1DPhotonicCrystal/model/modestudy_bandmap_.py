"""
STUDY_
Mode
band map
"""

import meep as mp
from meep import mpb
import numpy as np

import meep_mod.mod as mod
from meep_mod.gallery.study_ import StdStudy_Mode_


class ModeStudy_BandMap_(StdStudy_Mode_):
    def __init__(self, model, parity=mp.NO_PARITY, **kwargs):
        super(ModeStudy_BandMap_, self).__init__(model, **kwargs)
        # ---------- locals ----------
        # --- parity ---
        self.parity = parity
        # --- data ---
        self.k_points = None
        self.all_freqs = None
        self.eps = None
        # --- band funcs ---
        self._band_funcs = [mpb.output_efield] + [mpb.output_hfield]

    def run_study(self, ms: mpb.ModeSolver, output_field=True):
        self._create_dir()
        self.k_points = ms.k_points

        if output_field:
            ms.filename_prefix = self._runtime_dir + '/' + 'bandfield'  # overwrite output dir of ms
            ms.run_parity(self.parity, True, *self._band_funcs)
        else:
            ms.run_parity(self.parity, True)
        self.all_freqs = np.array(ms.all_freqs)

        md = mpb.MPBData(rectify=True, resolution=ms.resolution[0])
        self.eps = md.convert(ms.get_epsilon())

