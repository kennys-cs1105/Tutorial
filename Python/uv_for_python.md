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

