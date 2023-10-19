"""visualization"""

import numpy as np
import matplotlib.pyplot as plt
import meep_mod.mod as mod

study_dirname = "./data/FDTDstudy_2023-10-19_20-20-47"  # 结果文件夹
filename = study_dirname + "/result/model.h5"
result = mod.read_h5(filename)

fluxes = result['FDTD_flux_']['fluxes_history']
freqs = result['FDTD_flux_']['fluxes_freq']
t = result['FDTD_flux_']['t']

dfluxes_dt = fluxes[:,1:,1]-fluxes[:,:-1,1]

fig = plt.figure(figsize=[9,3])
axs = fig.subplots(1,2)
ax=axs[0]
ax.plot(fluxes[:,:,1].transpose())
ax.set_title('flux')
ax.set_xlabel('t')
ax.set_ylabel('P')
ax.grid(True)
ax=axs[1]
ax.plot(dfluxes_dt.transpose())
ax.set_title('$\partial_t$ flux')
ax.set_xlabel('t')
ax.set_ylabel('P')
ax.grid(True)
plt.show()

print('eta=',max(fluxes[1,:,1])/max(fluxes[0,:,1]))

# ----------------
study_dirnames = [
    "./data/scan_2023-10-19_20-23-58/FDTDstudy_2023-10-19_20-23-58_0",
    "./data/scan_2023-10-19_20-23-58/FDTDstudy_2023-10-19_20-24-37_1",
    "./data/scan_2023-10-19_20-23-58/FDTDstudy_2023-10-19_20-25-53_2",
]  # 结果文件夹
ls = []
etas = []
for study_dirname in study_dirnames:
    filename = study_dirname + "/result/model.h5"
    result = mod.read_h5(filename)

    l = result['wg_l']
    fluxes = result['FDTD_flux_']['fluxes_history']
    eta = max(fluxes[1, :, 1]) / max(fluxes[0, :, 1])

    ls.append(l)
    etas.append(eta)

fig = plt.figure(figsize=[5,3])
ax = fig.subplots(1,1)
ax.plot(ls, etas)
ax.set_title('$\eta$ vs l')
ax.set_xlabel('l')
ax.set_ylabel('$\eta$')
ax.grid(True)
plt.show()