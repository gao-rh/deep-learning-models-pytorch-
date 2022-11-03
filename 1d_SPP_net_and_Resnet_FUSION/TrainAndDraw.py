from ast import Global
from fileinput import filename
import time
import numpy as np
import pandas as pd
import torch
import torchvision
from IPython import display
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch import nn 
import os
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

class Timer():  #@save

    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    def __init__(
        self, xlabel=None, ylabel=None, legend=None, xlim=None, 
        ylim=None, xscale='linear', yscale='linear',
        fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, 
        figsize=(3.25, 2.5)
    ):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def accuracy(y_hat, y):
    """正确数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    """n个变量累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [ a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0, 0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算指定坐标集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric [1]

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

def evaluate_accuracy_gpu_fusion(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for X1, X2, y in data_iter:
            if isinstance(X1, list):
                # Required for BERT Fine-tuning (to be covered later)
                X1 = [x.to(device) for x in X1]
                X2 = [x.to(device) for x in X2]
            else:
                X1 = X1.to(device)
                X2 = X2.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X1, X2), y), size(y))
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, lr, device, model_name, isini=0):
    train_inf = []
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if not isini:
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X = X.to(device)
            y = y.type(torch.LongTensor) #add for debug
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            #if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
        #draw
        animator.add(epoch + (i + 1) / num_batches,
             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        #record
        #print(f'epoch:{epoch},  train:{train_acc:.3f}    test:{test_acc:.3f}')
        train_inf.append([epoch, train_acc, test_acc, train_l])
    df = pd.DataFrame(train_inf, columns=['epoch', 'train_accuracy', 'test_accuracy', 'loss'])
    #now = time.strftime("%Y%m%d", time.localtime())
    filename = './train_info/' + model_name +'.xlsx'
    try:
        df0 = pd.read_excel(filename)
        df = [df0, df]
        df = pd.concat(df)
        df.to_excel(filename)
    except:
        df.to_excel(filename)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}') 
def train_Fusion_2(net, train_iter, test_iter, num_epochs, lr, device, model_name, isini=1, isrecord=False):
    train_inf = []
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if not isini :
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X1, X2, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X1 = X1.to(device)
            X2 = X2.to(device)
            y = y.type(torch.LongTensor) #add for debug
            y = y.to(device)
            y_hat = net(X1, X2)[0]
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X1.shape[0], accuracy(y_hat, y), X1.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            #if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
        #draw
        animator.add(epoch + (i + 1) / num_batches,
             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu_fusion(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        #record
        #print(f'epoch:{epoch},  train:{train_acc:.3f}    test:{test_acc:.3f}')
        train_inf.append([epoch, train_acc, test_acc, train_l])
    df = pd.DataFrame(train_inf, columns=['epoch', 'train_accuracy', 'test_accuracy', 'loss'])
    #now = time.strftime("%Y%m%d", time.localtime())
    filename = './train_info/' + model_name +'.xlsx'
    try:
        df0 = pd.read_excel(filename)
        df = [df0, df]
        df = pd.concat(df)
        df.to_excel(filename)
    except:
        df.to_excel(filename)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}') 