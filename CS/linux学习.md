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