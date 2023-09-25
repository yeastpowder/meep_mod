"""
io methods
.h5 file is used here for data storage and recover
"""

import os
import inspect
import h5py as h5
import numpy as np
from .env import master_do, really_master_do

import meep as mp
import meep.mpb as mpb


# dir operation
def new_dir(name, path='.'):
    """
    创建目录，若目录已存在则跳过
    :param name: dir name
    :param path: create dir under path
    :return: full dir path
    """
    new = path + '/' + name

    @ master_do
    def _new_dir():
        if not os.path.exists(path):
            raise ValueError("path not exist")
        if os.path.exists(new):
            print("exist dir", new)
        else:
            os.mkdir(new)
            print("mkdir:", new)

    _new_dir()
    return new


#  模型参数 与 h5 数据格式 的转换
#  规则
#      class instance/namedtuple/list/tuple/dict -> h5 group
#      class/function/method/str/numerical -> h5 group_attr(str/numeric)
#      ndarray -> h5 dataset
def _get_instance_components(obj, keys=None):
    """
    instance of a class ->
        [('_am', 'class instance'),
        ('_class', class name),
        (key, member), ..., (key, member)]
    """
    obj_dict = obj.__dict__
    keys = keys if keys else obj_dict.keys()
    components = [('_am', 'class instance'),
                  ('_class', str(obj.__class__))]
    for key in keys:
        components.append((key, obj_dict[key]))
    return components


def _get_namedtuple_components(obj, keys=None):
    """
    instance of a nametuple ->       # e.g. a harminv.mode
        [('_am', 'namedtuple instance'),
        ('_class', class),
        (key, member), ..., (key, member)]
    """
    keys = keys if keys else obj._fields
    components = [('_am', 'namedtuple instance'),
                  ('_class', str(obj.__class__))]
    for key in keys:
        components.append((key, getattr(obj, key)))
    return components


def _get_dict_components(obj, keys=None):
    """
    a dict obj ->
        [('_am', 'dict'),
        (key, member), ..., (key, member)]
    """
    keys = keys if keys else obj.keys()
    components = [('_am', 'dict')]
    for key in keys:
        components.append((key, obj[key]))
    return components


def _get_iterable_components(obj, keys=None):
    """
    a iterable obj, i.e. list/tuple/... ->
        [('_am', 'iterable'),
        ('_len', len),
        (key, member), ..., (key, member)]
    """
    keys = keys if keys else range(len(obj))
    components = [('_am', 'iterable'),
                  ('_len', len(obj))]
    for key in keys:
        components.append((str(key), obj[key]))
    return components


def _get_info(obj):
    """
    class/func/method ->
        '<class/func/method -> ...>'
    """
    return str(obj)


def _get_obj(obj):
    """
    str/numeric ->
        str/numeric
    """
    return obj


def _get_none():
    """
    None ->
        'None'
    """
    return 'None'


def _recover_none():
    return None


def _get_bool(obj):
    """
    Bool ->
        'True'/'False'
    """
    return str(obj)


def _recover_bool(value):
    return value == 'True'


# .h5 IO method
@ master_do
def output_h5(obj, file, exceptions=[], exceptions_keys=[]):
    """
    把一个 meep模型(python obj) 下的参数写入 .h5 文件，跳过其中 种类为exception 或者 名称为exceptions_key 的子项目
    :param obj: object
    :param file: 'filename'
    :param exceptions: list: [exception]
    :param exceptions_keys: list: ['exceptions_key']
    :return:
    """
    exceptions += [mp.Simulation, mpb.ModeSolver]

    exceptions_keys += ['swigobj']      # some meep objects, i.e. Source, may contain a .swigobj and will cause problems
    exceptions_keys += ['model']      # prevent inf loop

    def _get_components(o):
        if inspect.isclass(o) or inspect.isfunction(o) or inspect.ismethod(o):  # class/func
            components = None
        elif hasattr(o, '__dict__'):  # class instance
            components = _get_instance_components(o)
        elif isinstance(o, np.ndarray):  # ndarray data
            components = None
        elif isinstance(o, dict):  # dict
            components = _get_dict_components(o)
        elif hasattr(o, '_fields'):  # namedtuple
            components = _get_namedtuple_components(o)
        elif isinstance(o, (list, tuple)):  # list/tuple
            components = _get_iterable_components(o)
        else:
            components = None
        return components

    def _get_value(o):
        if inspect.isclass(o) or inspect.isfunction(o) or inspect.ismethod(
                o):  # class/func/method
            value = _get_info(o)
        elif isinstance(o, str):  # str
            value = _get_obj(o)
        elif o is None:  # None
            value = _get_none()
        elif isinstance(o, bool):  # bool
            value = _get_bool(o)
        else:  # assume else is a numeric obj
            value = _get_obj(o)
        return value

    def _write(o, group):
        components = _get_components(o)
        if components:
            for key, sub_o in components:   # check all components
                # 1. solve exceptions
                if key in exceptions_keys:      # pass exception keys
                    pass
                elif exceptions and isinstance(sub_o, tuple(exceptions)):  # pass the exception classes
                    pass
                # 2. else write
                else:
                    # print(o)
                    if _get_components(sub_o):  # if sub_o is still a group
                        sub_group = group.create_group(key)
                        _write(sub_o, sub_group)
                    elif isinstance(sub_o, np.ndarray):   # if sub_o is a ndarray
                        group.create_dataset(key, data=sub_o)
                    else:   # if sub_o is a simple attr
                        value = _get_value(sub_o)
                        group.attrs[key] = value
        elif isinstance(o, np.ndarray):  # if o is a ndarray
            group.create_dataset('data', data=o)
        else:  # if o is a simple attr
            value = _get_value(o)
            group.attrs['value'] = value

    with h5.File(file, 'w') as f:
        _write(obj, f)
    return file


@ master_do
def read_h5(file, group=None):
    """
    读取 .h5 文件，可选择只读取其中的 group
    :param file: 'filename'
    :param group: choose a group
    :return: data
    """
    def _recover_value(v):
        if v == 'None':
            return _recover_none()
        elif v in ["True", "False"]:
            return _recover_bool(v)
        return v

    def _read(g):
        if isinstance(g, h5.Dataset):    # if g is a dataset
            return g[:]
        elif isinstance(g, h5.Group):       # if g is a group
            if g.attrs['_am'] == 'iterable':   # g is a list-like
                lst = []
                l = g.attrs['_len']
                for i in range(l):
                    if str(i) in g.keys():
                        lst.append(_read(g[str(i)]))    # g is a list of groups
                    elif str(i) in g.attrs.keys():
                        value = _recover_value(g.attrs[str(i)])
                        lst.append(value)     # g is a list of attrs
                return lst
            else:   # g is a dict-like
                dct = {}
                for key in g.keys():        # g is a dict of containers
                    dct[key] = _read(g[key])
                for key in g.attrs.keys():     # g is a dict of simples
                    value = _recover_value(g.attrs[key])
                    dct[key] = value
                return dct
        return None

    with h5.File(file, 'r') as f:
        if group:
            f = f[group]
        obj = _read(f)
    return obj

