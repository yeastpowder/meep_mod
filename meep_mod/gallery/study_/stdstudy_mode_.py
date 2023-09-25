"""
STANDARD STUDY_
Mode
"""

import meep as mp
from meep import mpb
import meep_mod.mod as mod


class StdStudy_Mode_(mod.FdtdStudy):
    def __init__(self, model,
                 output_dir=None, output_name=None,
                 _n=None):
        """
        Mode 研究，连接模式展开求解器，运行模拟并输出结果
        :param model: mod.Model: parent model
        :param output_dir: output study to output_dir/output_name_yy-mm-dd_hh-mm-ss, default as '.'
        :param output_name: output study to output_dir/output_name_yy-mm-dd_hh-mm-ss, default as 'Modestudy'
        :param _n: _n is an optional index to distinguish studies of different mpi-groups, cluster nodes, etc
        """
        super(StdStudy_Mode_, self).__init__(model)
        # ---------- locals ----------
        # -- env values --
        self._n = _n
        self._create_time = mod.get_time()    # when study is init, mpi processes use the time of rank0
        self._processor = mod.get_processor_name()
        # -- output dir --
        self.output_name = output_name if output_name else 'Modestudy'
        self.output_name = self.output_name + '_' + mod.get_time()
        self.output_dir = output_dir if output_dir else '.'
        self._study_dir = None
        self._result_dir = None
        self._runtime_dir = None
        # -- band funcs --
        self._band_funcs = []
        # -- data --
        pass

    def _create_dir(self):
        """创建输出目录"""
        if self._n is not None:
            self._study_dir = mod.new_dir(self.output_name + '_' + str(self._n), self.output_dir)
        else:
            self._study_dir = mod.new_dir(self.output_name, self.output_dir)
        self._result_dir = mod.new_dir('result', self._study_dir)
        self._runtime_dir = mod.new_dir('runtime', self._study_dir)

    def run_study(self, ms: mpb.ModeSolver):
        """
        接入Mode求解器 运行该 study 所有测试
        :param ms: mpb.ModeSolver
        """
        self._create_dir()
        mpb.ModeSolver.run(*self._band_funcs)

    @ mod.master_do
    @ mod.with_matplotlib
    def plot2D(self, ms: mpb.ModeSolver):
        """
        画出模型的 三个切面 二维图像
        :param ms: mpb.ModeSolver
        """
        import matplotlib.pyplot as plt

        md = mpb.MPBData(rectify=True, resolution=ms.resolution[0])
        eps = ms.get_epsilon()
        converted_eps = md.convert(eps)

        for ax in ['x', 'y', 'z']:
            fig, a = plt.subplots(1, 1)
            mod.plot_data_eps(fig, a, mod.center_slice(converted_eps, ax=ax), alpha=1)
            plt.show()

    @ mod.master_do
    def output_result(self):
        """将 Model/Study 数据写入 .h5，如果有父级 model 优先保存 model """
        if self.model:
            filename = "model.h5"
            mod.output_h5(self.model, self._result_dir + '/' + filename)
        else:
            filename = "study.h5"
            mod.output_h5(self, self._result_dir+'/'+filename)


# %% __main__ test
if __name__ == '__main__':
    import numpy as np

    a = 0.33
    w = 0.7
    h = 0.22
    ff = 0.2
    r = np.sqrt(a * w / np.pi)

    wg = [mp.Block(size=mp.Vector3(mp.inf, w, h), material=mp.Medium(epsilon=12))]
    hole = [mp.Ellipsoid(size=mp.Vector3(r, r, mp.inf))]

    unit_cell = mp.Vector3(a, 4, 3)
    ms = mpb.ModeSolver(resolution=50,
                        geometry_lattice=mp.Lattice(size=unit_cell),
                        geometry=wg+hole,
                        num_bands=3, k_points=mp.interpolate(10, [mp.Vector3(), mp.Vector3(x=0.5)]))
    ms.init_params(mp.NO_PARITY, True)

    test_ = StdStudy_Mode_(None)
    test_.plot2D(ms)

