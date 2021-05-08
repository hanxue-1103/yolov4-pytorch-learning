# 1. 安装环境
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

注意：
（1）ModuleNotFoundError: No module named 'mish_cuda'
把所有的```from mish_cuda import MishCuda as Mish```替换成
```python
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
```

# 2. 代码说明
参考：https://github.com/WongKinYiu/PyTorch_YOLOv4
代码参考上面的代码，本人从学习的角度去学习这个代码，再次基础上增加一些接口脚本。

cfg：存储模型的配置文件
data：存储数据集信息文件
models：模型构建的py文件
utils：相关函数的py文件
train.py：训练代码

## 2.1. 下面的代码是在原作者的基础上添加的方便调用。
在运行测试的时候需要对对应的路径进行更改（图片路径，cfg文件路径，模型权重路径，视频文件路径，以及图片输入大小等）。
我写的模型文件需要读取pth，作者保存的是pt结尾的文件，其实在训练的过程中更改一下保存方式就好，或者把predict_image.py文件下的init()函数里面的读取权重的方式更改一下就好。cfg文件的yolo层的classes数需要更改，同时更改上一卷积层的fitter的值=(classes+5)*3

predict_image.py：预测一张图片的代码。
predict_video.py：预测一段视频

