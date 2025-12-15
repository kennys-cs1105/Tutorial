# uv for python

*Created By KennyS*

---

## 安装uv

1. 安装uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. pip安装

```
pip install uv
```

## 工控机离线安装uv的方法

**手动下载二进制文件，无需脚本**

1. 在有网机器上下载uv二进制文件

```bash
# x86_64架构
wget https://github.com/astral-sh/uv/releases/download/0.4.14/uv-x86_64-unknown-linux-gnu -O uv
# ARM64架构
# wget https://github.com/astral-sh/uv/releases/download/0.4.14/uv-aarch64-unknown-linux-gnu -O uv
```

2. 传输到离线工控机

```bash
# 1. 传输二进制文件到离线机的/usr/local/bin（系统全局可执行目录）
scp uv root@离线机IP:/usr/local/bin/
# 2. 离线机赋予执行权限
chmod +x /usr/local/bin/uv
# 3. 验证（无需配置环境变量，/usr/local/bin默认在PATH中）
uv --version
```

3. 离线后使用uv

在有网机器上下载包的离线依赖

```bash
# 有网机器：下载包及依赖到本地目录
uv pip download nnunetv2 scipy SimpleITK --dest ./offline-packages/ # 目录下下载whl/tar.gz文件
# 导出依赖
uv pip freeze --from-downloads ~/uv-offline-packages > ~/uv-offline-requirements.txt
# 打包（压缩后传输更快）
tar -zcvf uv-offline-packages.tar.gz ~/uv-offline-packages ~/uv-offline-requirements.txt
# 示例：SCP传输到离线机（离线机需开启SSH）
scp uv-offline-packages.tar.gz root@离线机IP:/root/
```

在离线机器上解压安装

```bash
# 解压传输过来的包
tar -zxvf uv-offline-packages.tar.gz -C /root/
# 进入离线包目录
cd /root/uv-offline-packages
```

```bash
# 方式1
uv venv env-name
source env-name/bin/activate
uv pip install *** --no-index --find-links ./offline-packages/
# 方式2
uv pip install --no-index --find-links ./offline-packages/ -r uv-offline-requirements.txt
```

## uv配置

1. uv下载速度慢, 配置环境变量

```
# 推荐使用清华源
echo 'export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"'>> ~/.bashrc

# 或者用阿里源
# echo 'export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"' >> ~/.bashrc

# 让配置立即生效
source ~/.bashrc
```

2. 也可以在toml文件中定制

```
[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default=true
```

