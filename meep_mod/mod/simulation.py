"""mod of meep.simulation"""

import meep as mp
import numpy as np

from collections import namedtuple

# 单位与步长的换算
Unit = namedtuple("Unit", ["length_unit", "time_unit", "freq_unit"])


def unit(length_unit=None):
    """
    单位换算 meep->SI
    :param length_unit: 1->x m
    :return: "length_unit": 1->x m, "time_unit": 1->y s, "freq_unit": 1->z Hz
    """
    if length_unit is None:
        length_unit = 1e-6  # default length_unit a=1[um]

    length_unit = length_unit
    c_unit = 3e8

    time_unit = length_unit / c_unit
    freq_unit = c_unit / length_unit

    si_unit = Unit(length_unit=length_unit, time_unit=time_unit, freq_unit=freq_unit)
    return si_unit


Step = namedtuple("Step", ["length_step", "time_step", "freq_step"])


def step(resolution, f_cen):
    """
    FDTD网格分割（meep单位）
    :param resolution: sim.res
    :param f_cen: 中心频率
    :return: "length_step", "time_step", "freq_step"
    """

    length_step = 1 / resolution
    time_step = 0.5 * length_step / 1  # dt = courant_factor * dx / c
    freq_step = -1.0 * f_cen * f_cen * time_step  # df = d(1/T) = -1/T^2 dt = -f^2 dt

    mp_step = Step(length_step=length_step, time_step=time_step, freq_step=freq_step)
    return mp_step


# 构造 step funcs, 以在 FDTD时间演化 的计算过程中执行额外任务
def make_step_func(func, *args, **kwargs):
    """
    将一个（与simulation无关的）func改写成 step func 的基本形式
    """
    def _step_func(sim):
        func(*args, **kwargs)
    return _step_func


def make_step_func_sim(func, *args, **kwargs):
    """
    将一个（与simulation有关的）func改写成 step func 的基本形式
    """
    def _step_func(sim):
        func(sim, *args, **kwargs)
    return _step_func


def lists2ndarrays(obj, keys):
    """
    Step Function: 将 obj[key] 由list转变成ndarray
    :param obj:
    :param keys:
    :return: step_function()
    """
    def _list2ndarray(sim):
        for key in keys:
            temp = getattr(obj, key)
            temp = np.array(temp)
            setattr(obj, key, temp)
    return _list2ndarray


def appendlst(lst, val):
    """Step Function: list.append"""
    def _append_lst(sim):
        lst.append(val)
    return _append_lst


def poplst(lst, index):
    """Step Function: list.pop"""
    def _pop_lst(sim):
        lst.pop(index)
    return _pop_lst


def append_sim_t(lst):
    """Step Function: 获取FDTD当前时间 并append至lst"""
    def _append_sim_t_lst(sim):
        lst.append(sim.meep_time())

    return _append_sim_t_lst


def append_harminv_modes(harm, lst, mxbands=None):
    """Step Function: 获取harminv当前结果 并append至lst"""
    def _append_harminv_modes_lst(sim):
        lst.append(harm._analyze_harminv(sim, maxbands=mxbands if mxbands else 100))  # force harm to calculate

    return _append_harminv_modes_lst


def append_dft(dft_obj, lst, components, n_frq):
    """Step Function: 获取dft_obj(dft/flux/energy/force/near2far)当前结果 并append至lst"""
    def _append_dft_lst(sim):
        sub_lst = []
        for num_frq in range(n_frq):
            sub_lst.append([])
            for i, component in enumerate(components):
                sub_lst[num_frq].append(sim.get_dft_array(dft_obj=dft_obj, component=component, num_freq=num_frq))
        lst.append(np.array(sub_lst))

    return _append_dft_lst


def append_flux(flux, lst):
    """Step Function: 获取flux当前结果 并append至lst"""
    def _append_flux_lst(sim):
        lst.append(mp.get_fluxes(flux))

    return _append_flux_lst


def append_volume(volume, lst, components):
    """Step Function: 获取volume内电磁场当前结果 并append至lst"""
    def _append_volume_lst(sim):
        sub_lst = []
        for i, component in enumerate(components):
            sub_lst.append(sim.get_array(component=component, size=volume.size, center=volume.center))
        lst.append(sub_lst)
    return _append_volume_lst


def append_mode_coeff(flux, lst, bands, **kwargs):
    """Step Function: 获取电磁场模式分解当前结果 并append至lst"""
    def _append_mode_coeff(sim):
        lst.append(sim.get_eigenmode_coefficients(flux, bands, **kwargs))
    return _append_mode_coeff


def output_dft(fname, dft):
    """Step Function: 保存dft当前结果"""
    def _output_dft(sim):
        outdir = sim.fields.outdir
        prefix = sim.get_filename_prefix()
        t = sim.meep_time()
        mp.all_wait()   # it seems mp.save method have no problem withou this
        mp.Simulation.output_dft(sim, dft, outdir+'/'+prefix+'-'+fname+'-%09.2f' % t)
    return _output_dft


def save_flux(fname, flux):
    """Step Function: 保存flux当前结果"""
    def _save_flux(sim):
        t = sim.meep_time()
        mp.Simulation.save_flux(sim, fname+'-%09.2f' % t, flux)
    return _save_flux


def save_near(fname, flux):
    """Step Function: 保存nearfield当前结果"""
    def _save_near(sim):
        t = sim.meep_time()
        mp.Simulation.save_near2far(sim, fname+'-%09.2f' % t, flux)
    return _save_near


def save_energy(fname, flux):
    """Step Function: 保存energy当前结果"""
    def _save_energy(sim):
        t = sim.meep_time()
        mp.Simulation.save_energy(sim, fname+'-%09.2f' % t, flux)
    return _save_energy


def output_times():
    """Step Function: 保存时间当前结果"""
    def _output_times(sim):
        outdir = sim.fields.outdir
        prefix = sim.get_filename_prefix()
        mp.Simulation.output_times(sim, outdir+'/'+prefix+"times")
    return _output_times



