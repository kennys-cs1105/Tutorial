# 深度学习服务器配置相关

*Created by KennyS*

---

- [稚晖君个人服务器](https://zhuanlan.zhihu.com/p/336429888)

## Linux换源

1. 备份原来的源

`cp /etc/apt/sources.list /etc/apt/sources.list.bak`

2. 换源

- `sudo vim /etc/apt/sources.list`

- 阿里源
    ```
    deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
    ```

3. 更新

```
sudo apt update
sudo apt upgrade
```

4. python的pip源

```
sudo vim ~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/ 
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```


## CUDA安装

1. 确认NVIDIA显卡驱动

2. CUDA版本不超过NVIDIA最高版本

```
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

3. 安装cuda时取消驱动安装

4. 环境变量设置

```
nano  ~/.bashrc

export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

source ~/.bashrc
```


## anaconda安装

1. 最好选择miniconda

2. 下载miniconda.sh

```
chmod +x Anaconda.sh
./Anaconda.sh
conda init
```


## Torch安装

1. conda env

```
conda create -n env python=3.10 # 最好选择3.10以上版本
conda activate env
pip3 install torch torchvision torchaudio # conda或者pip安装
```


## CUDNN安装

1. CUDA12.1

2. cudnn-linux-x86_64-8.9.3.28_cuda12-archive

3. 安装

```
tar -xzvf cudnn-11.0-linux-x64-v8.0.5.39.tgz

# 自己的cuda路径
sudo cp cuda/lib64/* /usr/local/cuda-11.0/lib64/
sudo cp cuda/include/* /usr/local/cuda-11.0/include/

cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```


## Linux端硬盘分区、挂载等

参考: 
- [亚马逊云服务器端配置](https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/add-instance-store-volumes.html)

- [挂载配置](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-using-volumes.html#ebs-mount-after-reboot)


1. 使用 `df -h` 命令查看已格式化并挂载的卷

2. 使用 `lsblk` 查看在启动时已映射但未格式化和装载的所有卷

3. 格式化并挂载仅映射的实例存储卷

    - 创建文件系统: `sudo mkfs -t ext4 /dev/nvme1n1`, 服务器选择`ext4`而不是`xfs`
    - 创建要将设备挂载到的目录: `sudo mkdir /data`
    - 在新建目录上挂载设备: `sudo mount /dev/sdc /data`

4. 修改配置

    - 备份: `sudo cp /etc/fstab /etc/fstab.orig`
    - 查看id: `sudo blkid`
    - 修改: `sudo vim /etc/fstab` -> `UUID=aebf131c-6957-451e-8d34-ec978d9581ae  /data  xfs  defaults,nofail  0  2`

5. 验证: 

    ```
    sudo umount /data
    sudo mount -a
    ```

### Linux磁盘修复相关

1. 检查磁盘属性`Linux LVM`, 微软的总会报错
2. 如果磁盘出问题一些检查手段
    ```
    # 检查可读 可写, 文件系统有问题的话一般会变成read-only
    mount | grep "/data"

    # 挂载 修复
    sudo umount /data
    sudo fsck /dev/sdb1
    sudo mount /dev/sdb1 /data
    ```
3. 更改所有者
    ```sudo chown -R $USER:$USER /mnt/data```


## Linux安装sogou输入法

1. 添加语言`install` -> `Chinese Simplify`
2. 安装fcitx`sudo apt-get install fcitx`, 设置-语言选择添加到整个系统
3. 安装sogou`sudo dpkg -i sogoupinyin_版本号_amd64.deb`
4. 安装过程中有问题就`sudo apt -f install`
5. 安装输入法依赖
    ```
    sudo apt install libqt5qml5 libqt5quick5 libqt5quickwidgets5 qml-module-qtquick2
    sudo apt install libgsettings-qt1
    ```
6. 重启, 小键盘中`configuration`添加sogou


## clash安装与配置

### 使用客户端

[clash-verge-rev](https://github.com/clash-verge-rev/clash-verge-rev)

1. 选择对应操作系统的安装包进行安装
2. 获取clash配置文件导入并开启全局代理

### 在linux服务器端配置clash

1. 下载clash核心

```
cd /tmp
wget https://github.com/MetaCubeX/mihomo/releases/latest/download/mihomo-linux-amd64-v1.18.3.gz
gunzip mihomo-linux-amd64-v1.18.3.gz
sudo mv mihomo-linux-amd64-v1.18.3 /usr/local/bin/mihomo
sudo chmod +x /usr/local/bin/mihomo
```

2. 配置文件

```
mkdir -p ~/.config/clash
cd ~/.config/clash

wget -O config.yaml "https://你的新订阅链接"
```

3. 通过这种配置文件的方式可能会因为dns端口问题导致无法正常使用, 这里选择在clash桌面端生成的`config.yaml`复制到该路径下

4. 启动clash

临时启动

```
clash-meta -d ~/.config/clash
# 或
mihomo -d ~/.config/clash

# 后台运行
nohup clash-meta -d ~/.config/clash > clash.log 2>&1 &
# 或
nohup mihomo -d ~/.config/clash > clash.log 2>&1 &
```

设置systemd服务

```
sudo nano /etc/systemd/system/clash.service
```

添加内容

```
[Unit]
Description=Clash Proxy Service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/clash-meta -d /home/你的用户名/.config/clash
Restart=always

[Install]
WantedBy=multi-user.target
```

启用

```
sudo systemctl daemon-reload
sudo systemctl enable clash
sudo systemctl start clash
```