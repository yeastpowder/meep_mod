"""declarations of meep_mod ui sturcture"""


# Top structure
class Model(object):
    def __init__(self):
        pass


# A Model can have several Studies, which executes certain simulation tasks
class _Study(object):
    def __init__(self, model):
        self.model = model

    def run_study(self, *arg, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.run_study(*args, **kwargs)


# FDTD study
class FdtdStudy(_Study):
    def __init__(self, model):
        super(FdtdStudy, self).__init__(model)

    def run_study(self, sim):
        pass


#  Mode study
class ModeStudy(_Study):
    def __init__(self, model):
        super(ModeStudy, self).__init__(model)

    def run_study(self, ms):
        pass


# A Model can have several Nodes containing scripts to generate geometry, source, etc. Can be quoted by Studies
class _Node(object):
    def __init__(self, model):
        self.model = model

    def do(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.do(*args, **kwargs)


# geometry creator
class GeoScript(_Node):
    def __init__(self, model):
        super(GeoScript, self).__init__(model)
        self.do = self.create_geometry

    def create_geometry(self):
        pass


# sources creator
class SrcScript(_Node):
    def __init__(self, model):
        super(SrcScript, self).__init__(model)
        self.do = self.create_sources

    def create_sources(self):
        pass


# dft monitors creator
class DftMonScript(_Node):
    def __init__(self, model):
        super(DftMonScript, self).__init__(model)
        self.do = self.add_monitors

    def add_monitors(self, sim):
        pass


# volume monitor creator
class VolMonScript(_Node):
    def __init__(self, model):
        super(VolMonScript, self).__init__(model)
        self.do = self.create_volumes

    def create_volumes(self):
        pass
