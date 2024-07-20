# NeoVim 安装

*Created by KennyS*

---


## 环境

UBUNTU22.04


## 安装

1. 尽量下载较新版本的neovim, 低版本可能会与插件冲突

```bash
mkdir ~/nvim
cd nvim
wget https://github.com/neovim/neovim/releases/download/v0.9.5/nvim-linux64.tar.gz
tar -zxvf *.tar.gz
```


2. 环境变量设置

创建符号链接

```bash
cd /usr/bin
sudo ln -s ~/path/nvim-linux64/bin/nvim nvim
```


## 配置

1. Lazy Neovim

```bash
git clone https://github.com/LazyVim/starter ~/.config/nvim
rm ~/.config/nvim/.git
```


2. 自定义

- 创建```mkdir ~/.config/nvim```

- 配置 ```init.lua```以及其他配置