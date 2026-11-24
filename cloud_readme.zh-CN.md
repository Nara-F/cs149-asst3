# AWS 配置说明

本作业的性能测试需要在 Amazon Web Services（AWS）虚拟机实例上运行。我们将提供（或已经发放）可用于付费的学生代金券。下面介绍配置步骤。

> [!NOTE]
> 每天结束后请关闭实例，避免一夜之间耗尽额度！

## 创建虚拟机

1. 登录 [AWS EC2 控制台](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#Home)。在页面右上角确认当前区域与申请配额的区域一致，应为从 `us-west-2`、`ap-northeast-1`、`ap-northeast-2` 中随机选择的区域。
![切换区域](handout/switch-region.png?raw=true)

2. 点击 `Launch instance` 创建虚拟机实例。
![启动实例](handout/launch-instance.png?raw=true)

3. 点击 `Browse more AMIs`。
![搜索 AMI](handout/search-ami.png)

4. 在 `Community AMIs` 标签页搜索 `Deep Learning ARM64 Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20250613`。
![选择 AMI](handout/select-ami.jpeg)

5. 选择 `g5g.xlarge` 实例类型。
![实例类型](handout/instance-type.png?raw=true)

6. 将卷大小改为 65 GiB，以便安装作业所需软件包。
![存储](handout/storage.jpeg?raw=true)

7. 登录实例需要密钥对。在 `Key pair (login)` 中点击 `Create a new key pair` 并命名。浏览器会下载 `<key_name>.pem`，用它登录即将创建的虚拟机。之后即可启动实例。
![创建密钥对，第 1 步](handout/keypair-step1.png)
![创建密钥对，第 2 步](handout/keypair-step2.png)

8. 确认所有信息并启动实例。
![确认](handout/confirm-launch.jpeg)

9. 创建虚拟机后即可通过 **SSH** 登录。进入实例页面，点击 `Connect`，选择 SSH 标签页，查找公有 IPv4 DNS 名称。实例启动和分配 IP 可能需要一些时间。
![连接](handout/connect.png?raw=true)

按提示执行 `chmod 400 path/to/key_name.pem` 修改密钥权限。取得地址后，用以下命令登录：

~~~~
ssh -i path/to/key_name.pem ubuntu@<public_dns_name>
~~~~

> [!WARNING]
> 创建实例后，如果配置中途需要离开，请先关闭实例。持续运行可能耗尽额度并产生额外费用。

## 配置虚拟机环境

1. CUDA 应已默认安装。执行 `nvidia-smi` 检查版本，输出应类似下方示例。CUDA 版本应为 **12.8**，本作业使用的 GPU 是 **NVIDIA T4G**。

~~~~
$ nvidia-smi
Thu Oct 23 18:22:32 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.20             Driver Version: 570.133.20     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA T4G                     On  |   00000000:00:1F.0 Off |                    0 |
| N/A   53C    P8             13W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
~~~~

2. 执行以下命令安装作业需要的 FreeGlut 库：

~~~~
sudo apt-get update && sudo apt-get install freeglut3-dev -y
~~~~

3. 现在可以克隆 asst3 仓库并开始开发：

~~~~
git clone https://github.com/stanford-cs149/asst3.git
~~~~

## 从 AWS 取回代码

建议创建私有 Git 仓库并在其中开发，以降低代码丢失风险，同时保留历史版本。

也可在本地执行 `scp`，从远程机器取回文件：

~~~~
scp -i <path-to-your-private-key> lightsail-user@<instance-IP-addr>:/path/to/file /path/to/local_file
~~~~

> 译注：原文这条 `scp` 示例使用 `lightsail-user`，但上文 EC2 SSH 登录使用 `ubuntu`。从本说明创建的 Ubuntu 实例取回文件时，应将示例中的用户名替换为 `ubuntu`。

## 关闭虚拟机

使用结束后，在网页中停止实例，或在终端执行：

~~~~
sudo shutdown -h now
~~~~
