# 渲染个人网站

*Created by KennyS*

---

## js环境

1. 依赖
    - nodejs
    - npm

    ```bash
    sudo apt install nodejs
    sudo apt-get install npm
    ```

2. 查看环境版本

    ```bash
    nodejs -v 
    npm -v
    ```

3. 升级版本

    - 配置npm：`~/.npmrc`
    ```bash
    prefix=/mnt/d/software/Nodejs/node_global #安装目录
    cache=/mnt/d/software/Nodejs/node_cache # 缓存模块
    registry=http://registry.npm.taobao.org # npm镜像网站
    ```

    - 升级
    ```bash
    sudo npm install -g n
    sudo n stable
    npm install npm@latest -g
    ```

## 基于VuePress的个人网站模板

[仓库](https://github.com/liyupi/codefather/tree/template)

1. 核心文件
    - package.json
    - config.js


## 渲染个人网站、域名

### 内网穿透

1. 使外部能够访问

    [ngrok](https://ngrok.com/)

2. 本地安装，在网站中注册登录，并验证当前身份

3. 在js文件中`build`，得到端口号，基于`ngrok`渲染该端口
    `ngrok http 80`

4. 配置静态域名
    - `Cloud Edge/Domains`, 创建静态域名, 执行命令