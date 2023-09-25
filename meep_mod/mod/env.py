"""solving problems running under different environments"""

import meep as mp
import time
from platform import node


# 是否安装 matplotlib
def is_matplotlib_available():
    try:
        import matplotlib.pyplot as plt
        _avail = True
    except:
        _avail = False
    return _avail


def with_matplotlib(func):
    def _func(*args, **kwargs):
        if is_matplotlib_available():
            return func(*args, **kwargs)
        else:
            return False
    return _func


# 是否安装 mayavi
def is_mayavi_available():
    try:
        import mayavi
        _avail = True
    except:
        _avail = False
    return _avail


def with_mayavi(func):
    def _func(*args, **kwargs):
        if is_mayavi_available():
            return func(*args, **kwargs)
        else:
            return False
    return _func


# 是否在 MPI 下运行
def is_mpi_available():
    try:
        from mpi4py import MPI
        _avail = True
    except:
        _avail = False
    return _avail


def with_mpi(func):
    def _func(*args, **kwargs):
        if is_mpi_available():
            return func(*args, **kwargs)
        else:
            return False
    return _func


# decorator for MPI cluster
def master_do(do):
    """
    修饰器：MPI节点下运行时，仅master节点执行操作
    """
    def _master_do(*args, **kwargs):
        if is_mpi_available():
            if mp.am_master():
                return do(*args, **kwargs)
            else:
                return False
        else:
            return do(*args, **kwargs)
    return _master_do


def really_master_do(do):
    """
    修饰器：MPI节点下运行时，仅really_master节点执行操作
    """
    def _really_master_do(*args, **kwargs):
        if is_mpi_available():
            if mp.am_really_master():
                return do(*args, **kwargs)
            else:
                return False
        else:
            return do(*args, **kwargs)

    return _really_master_do


# communication between mpi processes
def gather_from_masters(data):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    master_ranks = mp.merge_subgroup_data(rank)

    data_list = []
    if rank in master_ranks:
        if rank != 0:
            comm.send(data, dest=0, tag=11)
        else:
            for r in master_ranks:
                if r == 0:
                    data_list.append(data)
                else:
                    data_list.append(comm.recv(source=r, tag=11))
    return data_list


# 运行环境信息
def get_time(sync=True):
    """
    获取当前时间(字符串)
    :param sync: 在MPI下是否只使用主节点时间
    :return: str: 'yy-mm-dd_hh-mm-ss'
    """
    now = time.localtime()
    if is_mpi_available() and sync:    # if run with MPI, use the time of rank 0
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        now = comm.bcast(now, root=0)

    return time.strftime(r"%Y-%m-%d_%H-%M-%S", now)


@with_mpi
def get_rank():
    """
    获取当前节点编号(字符串)
    :return: str: 'rank of node'
    """
    return "("+str(mp.my_rank())+")"


def get_processor_name():
    """
    获取当前处理器信息
    :return: list: [str: 'processor name']
    """
    if is_mpi_available():
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

        pname = str(MPI.Get_processor_name())
        pname_lst = comm.gather(pname, root=0)
        pname_lst = comm.bcast(pname_lst, root=0)
    else:
        pname = str(node())
        pname_lst = [pname]
    return pname_lst




