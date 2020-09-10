# Dog-Cat-Classification-keras
- 项目基于windows下啊完成，适合新入手学习MobileNet、ResNet等CNN网络的人员学习
## 1：下载数据集
- 默认存储路径位```D:\data\Dog-cat\train```，数据集位置：[dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)
## 2: 准备环境
- 本项目基于windows环境，使用anaconda3的python环境完成，用户需要自行安装.
- ```pip install -r requirements.txt```
## 3：开始训练
```python train.py```
## 4：测试
- 执行：```python test.py --help```自行了解参数详细用法，以下为示例.
- ```python test.py --model-path /PATH/TO/YOUR/MODEL/ --image-path /PATH/TO/YOUR/TEST/IMAGE/ --model 0 (or 1 or 2)```
