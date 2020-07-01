## IRkernel for Jupyter-notebook
- 为Jupyter配置R内核，在R中操作即可

### 1. 在R中安装IRkernel内核
- IRkernel内核用于使R语言具有交互能力
- `install.packages('IRkernel')`，安装交互式内核

### 2. 在R中启用与Jupyter的接口
- `IRkernel::installspec()`，允许当前用户调用交互式内核
- `IRkernel::installspec(user = FALSE)`，允许所有用户调用交互式内核
