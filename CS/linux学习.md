# Linux学习

*Created by KennyS*

[黑马程序员](https://www.bilibili.com/video/BV1n84y1i7td/?spm_id_from=333.337.search-card.all.click&vd_source=f007ff194d374ddfb484e469e9ee2e1f)

---


## 第一章 01. 操作系统概述

1. 硬件和软件
    - 计算机由硬件、软件组成
    - 硬件: 计算机系统中由电子、机械、光电元件等组成的各种物理装置的总称
    - 软件: 用户和计算机硬件之间的接口和桥梁

2. 操作系统
    - 用户和计算机硬件之间的桥梁, 调度和管理计算机硬件进行工作


## 第一章 02. 初识Linux

1. Linux内核

- [linux内核](www.kernel.org)

- Linux系统组成：
    - Linux系统内核: 调度cpu, 调度内存, 调度文件系统, 调度网络通讯, 调度IO
    - 系统级应用程序: 文件管理器, 任务管理器等
    
2. Linux发行版

- 任何人都可以获得并修改内核, 并且自行集成系统级程序
- 提供内核+系统级程序的完整封装 -> Liunx发行版

3. 虚拟机快照

- 避免损坏linux操作系统
- 通过快照将当前虚拟机的状态保存下来 


## 第二章 01. Linux目录结构

1. Linux目录结构

- 树形结构
- Linux没有盘符概念，只有一个根目录，所有文件都在它下面

2. Linux路径的描述方式

- Linux系统中，路径之间的层级关系用`/`表示
- windows系统中，路径之间的层级关系用`\`来表示


## 第二章 02. 命令入门

1. 什么是命令

- 操作指令
- 基础格式：`command [-options] [parameters]`
    - `ls -l /home/test`
    - `cp -r test1 test2`

2. ls命令

- `ls [-a -l -h] [path]`

3. ls命令拓展

- ls命令参数
    - 查看隐藏内容 `ls -a`
    - 查看大小 `ls -lh`



---

## Linux相关

1. `tar cf - * | ssh -p 35394 root@region-1.autodl.com "cd /root/autodl-tmp && tar xf -"`

- 这条命令用于将当前目录下的所有文件和子目录打包并通过 SSH 传输到远程服务器上，然后在远程服务器上解压缩这些文件

- tar cf - *
  - tar：tar 是一个用于归档文件的命令。
  - c：创建一个新的归档文件。
  - f -：指定归档文件的名字为 -，表示将归档输出到标准输出（stdout），而不是文件。
  - *：表示当前目录下的所有文件和子目录。
  - 这部分命令将当前目录下的所有文件和目录打包成一个归档流，并将其输出到标准输出（即将归档流传递给管道）。

- |
  - |：管道操作符，将前一个命令的标准输出作为下一个命令的标准输入。

- ssh -p 35394 root@region-1.autodl.com "cd /root/autodl-tmp && tar xf -"
  - ssh：用于通过 SSH 协议连接到远程主机。
  - -p 35394：指定 SSH 连接使用的端口号，这里是 35394。
  - root@region-1.autodl.com：连接到名为 region-1.autodl.com 的远程主机，并以 root 用户身份登录。
  - "cd /root/autodl-tmp && tar xf -"：在远程主机上执行的命令。
  - cd /root/autodl-tmp：更改当前工作目录到 /root/autodl-tmp。
  - tar xf -：解压从标准输入（stdin）读取的归档文件。x 是解压缩的选项，f - 表示从标准输入读取归档数据。