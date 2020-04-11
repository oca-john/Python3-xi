# Biopython(py3)-Seq

## Seq命令注意事项
`my_seq = Bio.Seq.Seq('ACTGGTAGTCAG')`展示Seq命令的真实目录树关系。
- Bio是Python3的一个扩展库Biopython的缩写（文件夹），Seq是其中操作序列的模块（py脚本文件），Seq是其中用于生成序列的命令（代码块）。

## 1.直接调用Seq命令
`from Bio.Seq import Seq`, 从Bio库的Seq模块（文件）中导入Seq命令（代码块）。
`my_seq = Seq('ATCG')`, 直接用Seq命令生成序列。

## 2.调用整个模块
`import Bio.Seq`, 调用Bio库的Seq模块（脚本文件）。
`my_seq = Bio.Seq.Seq('ATCG')`, 通过逐级调用Seq命令生成序列。
