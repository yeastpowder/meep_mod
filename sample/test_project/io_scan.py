"""test_project/io_scan.py"""

import meep as mp

import sys
sys.path.append("../..")  # help finding meep_mod
import meep_mod.mod as mod
from model import MainModel

mod.new_dir('data')
now = mod.get_time()
scan_dir = mod.new_dir('./data/scan_'+now)

for i, wg_l in enumerate([5,10,20]):
    model = MainModel(wg_l=wg_l,
                      symmetries=[mp.Mirror(mp.Y, phase=-1), mp.Mirror(mp.Z, phase=1)], dt=1,
                      output_dir=scan_dir, _n=i)

    study = model.FDTD_flux_
    sim = model.sim

    study.run_study(sim, output_field=False, until_after_sources=80)
    study.output_result()
