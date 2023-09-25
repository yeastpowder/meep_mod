"""

"""

import meep as mp

import sys
sys.path.append("../..")  # help finding meep_mod
import meep_mod.mod as mod
from model import MainModel

mod.new_dir('data')

model = MainModel(parity=mp.TE+mp.ODD_Y, n_b=3)
study = model.Mode_bandmap_
ms = model.ms
ms.geometry = model.wg + model.etch

study.run_study(ms, output_field=True)
study.output_result()

