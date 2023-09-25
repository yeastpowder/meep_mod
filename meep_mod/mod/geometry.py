"""mod of meep.geometry"""

import meep as mp
import numpy as np


# instance arrays
def vector3_array(n_3_array):
    """
    用 坐标阵列 生成 mp.Vector3 阵列
    :param n_3_array: ndarray n*3
    :return: [mp.Vector3]
    """
    n_3_array = np.array(n_3_array)
    shape = n_3_array.shape
    if len(shape) == 2 and shape[1] == 3:
        pass
    else:
        raise ValueError("wrong array shape, should be n*3")

    return list(map(lambda xyz: mp.Vector3(*xyz), n_3_array))


def vector3_array_xyz(x=0, y=0, z=0):
    """
    用 xyz阵列 生成 mp.Vector3 阵列
    :param x: []*n / float
    :param y: []*n / float
    :param z: []*n / float
    :return: [mp.Vector3]
    """
    def _get_len(ax):
        try: l_ax = len(ax)
        except: l_ax = 1
        return l_ax

    def _fix_len(ax, l):
        if _get_len(ax) == l:
            return ax
        elif _get_len(ax) == 1:
            return [ax]*l
        else:
            raise ValueError("shape not match for x,y,z")

    l_xyz = [_get_len(ax) for ax in [x,y,z]]
    l = max(l_xyz)
    x = _fix_len(x, l)
    y = _fix_len(y, l)
    z = _fix_len(z, l)

    return vector3_array(np.array([x,y,z]).transpose())

