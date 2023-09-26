# meep_mod

这是一个基于[meep](https://github.com/NanoComp/meep)的ui修改，在这里我根据一些主流的仿真软件使用习惯提供了一套模块化的模型构建方案。

## 安装
### 本地pip安装
可以直接pip安装本地包：\
下载文件 或者 git到本地 后在本文件夹内 ```pip install .```
### 配置环境
或者如果你想手动配置运行环境：
- [安装 meep 环境](https://meep.readthedocs.io/en/latest/Installation/)
```conda create -n mp -c conda-forge pymeep pymeep-extras```

或者 MPI：```conda create -n pmp -c conda-forge pymeep=*=mpi_mpich_*```

- 下载 meep_mod
直接下载文件 或者 git到本地

- 安装可视化环境
在这里的例子中，我使用了一些常用的package已完成数据可视化。

[jupyter lab](https://jupyter.org/) ``` pip install jupyterlab```

[matplotlib](https://matplotlib.org/) ``` pip install matplotlib```

[pandas](https://pandas.pydata.org/) ``` pip install pandas```

[vega-altair](https://altair-viz.github.io/) ``` pip install altair vega_datasets```

## 构建一个模型
本案例的模型结构如下：

*待补全*

## 案例
./sample文件夹下提供了一个分析硅基一维光子晶体腔的案例（*待补全*），详细说明参见[sample.ipynb](sample.ipynb)
