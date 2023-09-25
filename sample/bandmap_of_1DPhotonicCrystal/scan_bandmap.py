"""

"""

import meep as mp
import numpy as np

import sys
sys.path.append("../..")  # help finding meep_mod
import meep_mod.mod as mod
from model import MainModel

mod.new_dir('data')
outputdir = mod.new_dir('scan_Modestudy_'+mod.get_time(), path='./data')

ffs = np.linspace(0.02, 0.2, 10)
# a_s = [0.33]
a_s = np.linspace(0.3, 0.37, 8)

_n = 0
for a in a_s:   # 在cluster环境下该循环可以写成多个模型并行的形式
    for ff in ffs:
        model = MainModel(ff=ff, a=a,
                          parity=mp.TE+mp.ODD_Y, n_b=2, _n=_n, output_dir=outputdir)
        study = model.Mode_bandmap_
        ms = model.ms
        ms.geometry = model.wg + model.etch

        study.run_study(ms, output_field=False)
        study.output_result()

        _n += 1

