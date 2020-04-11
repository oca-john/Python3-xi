# Biopython(py3)-Seq

## Seq命令注意事项
`my_seq = Bio.Seq.Seq('ACTGGTAGTCAG')`展示Seq命令的真实目录树关系，但事实上不能这样调用。
- Bio是Python3的一个扩展库Biopython的缩写（文件夹），Seq是其中操作序列的模块（py脚本文件），Seq是其中用于生成序列的命令（代码块）。

## Seq命令正确调用
`from Bio.Seq import Seq`, 从Bio库的Seq模块（文件）中导入Seq命令（代码块）。
