# 计算机、网络、操作系统

*Created by KennyS*

---

## Linux端硬盘分区、挂载等

[亚马逊云服务配置](https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/add-instance-store-volumes.html)
[挂载配置](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-using-volumes.html#ebs-mount-after-reboot)

1. 使用 `df -h` 命令查看已格式化并挂载的卷
2. 使用 `lsblk` 查看在启动时已映射但未格式化和装载的所有卷
3. 格式化并挂载仅映射的实例存储卷
    - 创建文件系统: `sudo mkfs -t ext4 /dev/nvme1n1`, 服务器选择ext4而不是xfs
    - 创建要将设备挂载到的目录: `sudo mkdir /data`
    - 在新建目录上挂载设备: `sudo mount /dev/sdc /data`
4. 修改配置
    - 备份: `sudo cp /etc/fstab /etc/fstab.orig`
    - 查看id: `sudo blkid`
    - 修改: `sudo vim /etc/fstab -> UUID=aebf131c-6957-451e-8d34-ec978d9581ae  /data  xfs  defaults,nofail  0  2`
5. 验证: 
    - `sudo umount /data     sudo mount -a`


### 挂载u盘

1. 在wsl下挂载u盘, linux下应该也一样
2. wsl环境下, 挂载的u盘是win系统下, 在win下使用powershell查看`wmic diskdrive list brief
`
3. 确认u盘挂载位置, wsl下挂载的硬盘都在`/mnt`目录下, 查看`ls /mnt`
4. 创建挂载点并挂载
    ```
    sudo mkdir /mnt/usb
    sudo mount -t drvfs E: /mnt/usb
    ```
5. 卸载u盘`sudo umount /mnt/usb`


### 根目录挂载

1. 通常是将更大的硬盘挂载到根目录下
```bash
sudo mkdir /data
sudo mount /dev/sdb /data
```
2. 修改`/etc/fstab`
```bash
sudo blkid -> /dev/sdb UUID
nano /etc/fstab
UUID=XXX /data ext4 defaults 0 2
```

---

## Linux操作系统的一些东西

### 杀死进程

1. 服务器卡，训练慢，终端断过链接，无法手动终止训练
    - 解决：
        - 查找python相关进程：ps aux | grep python，找到想要停止的命令、进程、PID
        - 如果是占用了单个进程，可以kill -9 <PID>
        - 但大部分情况是多进程
        `#pkill 可以根据命令名终止所有相关进程`
        `pkill -f train.py`

    - 或者
    ```bash
    # ps 或 pgrep 找到主进程的PID
    ps aux | grep train.py
    pgrep -f train.py
    # 记下主进程的PID。
    kill -9 <PID>
    ```
    - 或者
    ```
    # xargs 结合 ps 一次性终止多个进程
    # 列出所有相关进程并终止
    ps aux | grep train.py | awk '{print $2}' | xargs kill -9
    ```

### 换了阿里源后gcc没有stdlib，依赖冲突无法安装g++

1. 换回ubuntu的默认源
    ```bash
    # /etc/apt/sources.list
    deb http://archive.ubuntu.com/ubuntu/ jammy main restricted
    deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted
    deb http://archive.ubuntu.com/ubuntu/ jammy universe
    deb http://archive.ubuntu.com/ubuntu/ jammy-updates universe
    deb http://archive.ubuntu.com/ubuntu/ jammy multiverse
    deb http://archive.ubuntu.com/ubuntu/ jammy-updates multiverse
    deb http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse
    deb http://archive.canonical.com/ubuntu jammy partner
    ```

2. 更新

    `sudo apt update`
    `sudo apt upgrade`

3. 修复损坏的包
    `sudo apt --fix-broken install`

4. 清理和重新安装
    `sudo apt clean`
    `sudo apt update`

5. 检查是否持有的包
    `sudo apt-mark showhold`
    `sudo apt-mark unhold <package-name>`

6. 强制重新安装核心库
    `sudo apt install --reinstall libc6 libc6-dev libc-dev-bin`
    `sudo apt install g++`

### 虚拟机快照
- 通过快照保存状态，以后通过快照恢复虚拟机

---

## 压缩

1. `gzip: stdin: decompression OK, trailing garbage ignored`
    - 解决：报这样的错通常是因为tar.gz文件的尾部有一串0x00或者0xff，这是由于很多场合下压缩算法都会在压缩完成后补充一些字节以对齐数据块。gzip在正确解压完tar.gz的内容后开始解压这样的全零填充字节就会报这样的错，并不会影响使用
    - 用安静模式 (gzip -q) 可以消除这些警报