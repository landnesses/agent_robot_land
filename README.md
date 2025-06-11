# 机器人智能控制框架

让人工智能自主获取环境数据并且生成机器人的动作帧

与此同一批次开发的另一个方向(我在公司的实习项目)：[基于fastgpt工作流框架下的智能体构建](https://github.com/tianbot/tianbot_ai_agents)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)




## 项目简介


 本项目在FastGpt框架下构建AI智能体，能够实现Agent自主获取环境数据、决策、并执行动作


## 功能特点

- [x] 支持多种构型机器人，可自定义代码知识库
- [x] 支持局域网/内网穿透远程传输人工智能编写的动作帧
- [x] 使用fastgpt框架下的图形化智能体搭建
- [x] 可令人工智能自主执行流程，无需人类参与

## 安装与依赖

```bash
# 克隆仓库
git clone https://github.com/tianbot/tianbot_ai_agents.git

# 使用docker 安装FastGpt框架
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
systemctl enable --now docker
# 安装 docker-compose
curl -L https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
# 验证安装
docker -v
docker-compose -v
# 创建文件夹并下载对应的docker compose
mkdir fastgpt
cd fastgpt
#可以使用项目文件夹中的yml与config文件，也可以去FastGpt官方库找最新的版本（由于不同版本使用了不同的渠道端口，所以会在模型配置上有少许不同）
#在docker中启动
sudo docker-compose up -d

#在浏览器中的localhost:3000端口可以看到相应界面
首次运行，会自动初始化 root 用户，密码为 1234（与环境变量中的DEFAULT_ROOT_PSW一致），日志可能会提示一次MongoServerError: Unable to read from a snapshot due to pending collection catalog changes;可忽略。
```
之后便可以配置人工智能api
详细配置可查看 FastGPT 官方文档：[点击查看文档](https://doc.fastgpt.cn/docs/)

注：由于网络环境原因，docker官方源无法访问，需要使用镜像站或者特定的上网手段

## 部署工作流
1.首先配置通信后端，这里以oneapi为例，登录oneapi，在localhost:3001页面，使用name:root passwd:123456 登录，之后使用 云端api/本地模型 进行渠道的配置，再生成相应的令牌，这样FastGpt就可以使用生成的令牌与llm通信了  
2.使用name:root passwd:1234 然后登录fastgpt后，进入工作台，在右上角点击新建工作流，点击进入一个空白工作流，在左上角名字旁边的拓展栏中选择导入json，将下载下来的channel.json 导入后，便可以调试、编辑工作流。  




## 整体项目架构
本项目分为以下四个模块  
1.视觉处理模块:通过图像识别+深度摄像头 获取物体相对于摄像头坐标系的三维空间坐标  
  
2.tf树转换模块：通过订阅tf话题将物体转换到base_link的相对坐标位置  
  
3.智能体模块：通过使用基于Fastgpt的Agent工作流框架，让大语言模型基于人类的自然语言命令+传感器获取的信息协作生成 子任务列表+待执行任务帧  
  
4.机械臂执行模块：使用神经网络+迭代器 神经网络输入物体位置--输出预估的六个关节角，达到目标附近的位置 然后使用迭代器迭代到准确位置  

##

## 项目目录结构

| 文件夹 | 描述 |
| ------ | ---- |
| `core` | 机械臂与机器人控制相关的示例脚本和测试代码，包含基于 MoveIt 的控制接口以及 HTTP 通信等实现 |
| `docker` | Docker 环境配置与 compose 文件，用于快速部署依赖环境 |
| `json` | FastGpt 工作流的示例 JSON 配置文件 |
| `neuro` | 机械臂逆运动学神经网络模型及训练、验证脚本 |
| `tf` | ROS TF 坐标变换相关的节点与工具脚本 |
| `vision` | 视觉识别模块，集成了 YOLOv5 等目标检测代码 |

## 使用说明
1.所有的代码文件夹下都有requirements.txt描述各自所需环境

2.vision中



