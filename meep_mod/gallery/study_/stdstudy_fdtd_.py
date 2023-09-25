"""
STANDARD STUDY_
FDTD
"""

import meep as mp
import meep_mod.mod as mod


class StdStudy_FDTD_(mod.FdtdStudy):
    def __init__(self, model, dft_monitors_=[], vol_monitors_=[],
                 output_dir=None, output_name=None,
                 _n=None,):
        """
        FDTD研究，连接FDTD求解器，运行模拟并输出结果
        :param model: mod.Model: parent model
        :param dft_monitors_: [mod.DftMonScript]
        :param vol_monitors_: [mod.VolMonScript]
        :param output_dir: output study to output_dir/output_name_yy-mm-dd_hh-mm-ss, default as '.'
        :param output_name: output study to output_dir/output_name_yy-mm-dd_hh-mm-ss, default as 'FDTDstudy'
        :param _n: _n is an optional index to distinguish studies of different mpi-groups, cluster nodes, etc
        """
        super(StdStudy_FDTD_, self).__init__(model)
        # ---------- locals ----------
        # -- monitors --
        self.dft_monitors_ = dft_monitors_
        self.vol_monitors_ = vol_monitors_
        self._dfts = []
        self._vols = []
        # -- env values --
        self._n = _n
        self._create_time = mod.get_time()    # when study is init, mpi processes use the time of rank0
        self._processor = mod.get_processor_name()
        # -- output dir --
        self.output_name = output_name if output_name else 'FDTDstudy'
        self.output_name = self.output_name + '_' + mod.get_time()
        self.output_dir = output_dir if output_dir else '.'
        self._study_dir = None
        self._result_dir = None
        self._runtime_dir = None
        # -- step funcs --
        self._step_funcs = []
        # -- data --
        pass

    # run methods
    def _create_dfts(self, sim):
        """向simulation添加当前study下所有 DFT monitor"""
        self._dfts = []
        for dft_ in self.dft_monitors_:
            self._dfts += dft_.add_monitors(sim)
        return self._dfts

    def _create_vols(self):
        """向simulation添加当前study下所有 Volume monitor"""
        self._vols = []
        for vol_ in self.vol_monitors_:
            self._vols += vol_.create_volumes()
        return self._vols

    def _create_dir(self):
        """创建输出目录"""
        if self._n is not None:
            self._study_dir = mod.new_dir(self.output_name + '_' + str(self._n), self.output_dir)
        else:
            self._study_dir = mod.new_dir(self.output_name, self.output_dir)
        self._result_dir = mod.new_dir('result', self._study_dir)
        self._runtime_dir = mod.new_dir('runtime', self._study_dir)

    def run_study(self, sim: mp.Simulation, **kwargs):
        """
        接入FDTD求解器 运行该 study 所有测试
        :param sim: mp.Simulation
        :param kwargs: kwargs for sim
        """
        self._create_dir()
        self._create_dfts(sim)
        self._create_vols()
        mp.Simulation.run(sim,
                          *self._step_funcs,
                          **kwargs)

    @ mod.master_do
    @ mod.with_matplotlib
    def plot2D(self, sim):
        """
        画出模型的 三个切面 二维图像
        :param sim: mp.Simulation
        """
        import matplotlib.pyplot as plt
        self._create_dfts(sim)
        for ax in ['x', 'y', 'z']:
            fig, a = plt.subplots(1, 1)
            mod.plot2D(sim, fig, a, volumes=self._create_vols(), output_plane=mod.center_plane(sim.cell_size, ax=ax))
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

    def checkpoint_dump(self, sim: mp.Simulation):
        """
        保存当前模拟到 checkpoint
        do not use a @ master decorator for meep chunks parallel write the file
        NOTICE: this is a new feature supported by meep ver 1.21+
        """
        dirname = "checkpoint"
        sim.dump(self, self._result_dir+'/'+dirname)

    def checkpoint_load(self, sim: mp.Simulation, from_dir):
        sim.load(from_dir)


# %% __main__ test
if __name__ == '__main__':
    from meep_mod.gallery.monitor_ import StdMon_Flux_, StdMon_Vol_

    test_ = StdStudy_FDTD_(None, dft_monitors_=[StdMon_Flux_(None, center=mp.Vector3(), size=mp.Vector3(0, 0.5, 0.5),
                                                             f_cen=0.6, f_width=0.1)],
                           vol_monitors_=[StdMon_Vol_(None, center=mp.Vector3(), size=mp.Vector3(1, 1, 1))])

    sim = mp.Simulation(cell_size=mp.Vector3(1, 1, 1), resolution=10,
                         geometry=[mp.Block(size=mp.Vector3(.5, .5, .5), material=mp.Medium(epsilon=12))],
                         sources=[mp.Source(mp.GaussianSource(frequency=0.6, fwidth=0.1), component=mp.Ey,
                                            center=mp.Vector3(), size=mp.Vector3())]
                         )

    test_.plot2D(sim)


