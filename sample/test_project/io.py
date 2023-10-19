"""test_project/io.py"""

import meep as mp

import sys
sys.path.append("../..")  # help finding meep_mod
import meep_mod.mod as mod
from model import MainModel

mod.new_dir('data')

model = MainModel(symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=1)], dt=1)

study = model.FDTD_flux_
sim = model.sim

study.run_study(sim, output_field=False, until_after_sources=50)
study.output_result()