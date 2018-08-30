# 实现一个简单的决策树模型训练预测界面(打包exe)
前面写了几个机器学习算法模型，想要做一个可视化的界面，方便数据导入和训练，经过多方调查，选择了比较好入门的wxPython来做GUI。(python版本是3.6.3)

[TOC]

## 一、安装GUI工具wxPython  
[wxPython下载地址](https://pypi.org/project/wxPython/)  
我用的是anaconda，因此将下载好的whl文件拷贝至anaconda安装位置下的Scripts文件夹(\Anaconda\Scripts)，在该文件夹下，shift+右键，打开shell窗口。 
>\>pip3 install wxPython-4.0.3-cp36-cp36m-win_amd64.whl

## 二、安装打包工具Pyinstaller  
>\>pip3 install pyinstaller

## 三、界面代码  
模型用的是我前面写过的决策树和绘制决策树，绘制决策树中有稍作改动  
[决策树模型代码](https://github.com/htshinichi/ML_model/tree/master/DecisionTree)  
### 1.导入库  
**注意：**虽然没用到sip，但是在打包中需要用到，否则报错"ModuleNotFoundError: No module named 'PyQt5.sip'"  
[问题解决参考](https://blog.csdn.net/yueguangMaNong/article/details/81139224)


```python
from PyQt5 import sip
import re
import os
import wx
import pandas as pd
import DecisionTREE
import DrawDecisionTREE 
```

### 2.初始化函数__init__(self,parent,title)  


```python
def __init__(self, parent, title):
    ####初始化界面####
    super(Mywin, self).__init__(parent, title = title,size = (800,600))
    ####定义面板窗口####
    panel = wx.Panel(self)
```

#### 2.1字体设置  
wx.Font(pointSize,family,style,weight,underline,faceName,encoding)　　[各参数含义](https://blog.csdn.net/u014647208/article/details/78486370)


```python
    ####字体设置,(大小、样式等)####
    font_title = wx.Font(30, wx.ROMAN, wx.NORMAL, wx.BOLD)#标题    
    font_btn = wx.Font(15, wx.ROMAN, wx.ITALIC, wx.NORMAL)#按键
    font_text = wx.Font(15,wx.ROMAN,wx.NORMAL,wx.NORMAL)#文本
    font_hint1 = wx.Font(10,wx.ROMAN,wx.NORMAL,wx.NORMAL)#提示1，用于操作提示/警告
    font_hint2 = wx.Font(15,wx.ROMAN,wx.NORMAL,wx.NORMAL)#用于显示预测结果
```

#### 2.2文本设置  
父类是panel，即在panel窗口上布置。  
pos用于设置位置，size用于设置大小。  
SetForegroundColour设置文本颜色，SetBackgroundColour设置文本背景颜色，SetFont设置文本大小


```python
    ####固定文本设置####      
    self.title = wx.StaticText(panel,-1,"决策树",pos=(650,0),size=(80,30),style=wx.ALIGN_RIGHT)#右对齐
    self.title.SetForegroundColour((255,0,255))#设置文本颜色
    self.title.SetBackgroundColour((255,255,0))#设置文本背景颜色 
    self.title.SetFont(font_title)

    self.author = wx.StaticText(panel,-1,"author:htshinichi--https://github.com/htshinichi",pos=(480,535),size=(120,20),style=wx.ALIGN_LEFT)#pos用于设置位置、size用于设置大小
    self.author.SetForegroundColour((240,128,128))#设置文本颜色
    self.author.SetFont(font_hint1)

    self.lbl1 = wx.StaticText(panel,-1,"选择划分算法进行训练",pos=(50,50),style=wx.ALIGN_LEFT)#pos用于设置位置、size用于设置大小      
    self.lbl1.SetFont(font_text)

    self.hint1 = wx.StaticText(panel,-1,"",pos=(410,50),style=wx.ALIGN_LEFT)
    self.hint1.SetFont(font_hint1)

    self.hint2 = wx.StaticText(panel,-1,"",pos=(510,490),style=wx.ALIGN_LEFT)
    self.hint2.SetFont(font_hint2)
```

#### 2.3输入栏设置  


```python
    ####输入文本设置####
    self.file = wx.TextCtrl(panel,-1, pos=(50,10),size=(200,20))#用于输入文件所在路径
    self.file.Bind(wx.EVT_TEXT,self.GetFilePath)

    self.sample = wx.TextCtrl(panel,-1, pos=(200,480),size=(200,50))#用于输入测试样本
    self.sample.Bind(wx.EVT_TEXT,self.GetSamplVector)
```

#### 2.4按键设置


```python
    ####按键设置####
    self.btn = wx.Button(panel,-1,"Select",pos=(260,10),size=(80,20)) #用于选择文件，可以不用自己输入路径了^^
    self.btn.Bind(wx.EVT_BUTTON,self.OpenFileDialog)
    self.btn.SetFont(font_btn)

    self.btn = wx.Button(panel,-1,"LoadData",pos=(350,10),size=(90,20)) #加载数据集
    self.btn.Bind(wx.EVT_BUTTON,self.LoadData)
    self.btn.SetFont(font_btn)

    self.btn = wx.Button(panel,-1,"Train",pos=(330,45),size=(50,30)) #训练模型
    self.btn.Bind(wx.EVT_BUTTON,self.Train)
    self.btn.SetFont(font_btn)  

    self.btn = wx.Button(panel,-1,"PlotDT",pos=(50,480),size=(80,50)) #显示训练好的决策树
    self.btn.Bind(wx.EVT_BUTTON,self.Plot)
    self.btn.SetFont(font_btn) 

    self.btn = wx.Button(panel,-1,"Predict",pos=(420,480),size=(80,50)) #预测类别
    self.btn.Bind(wx.EVT_BUTTON,self.GetPrediction)
    self.btn.SetFont(font_btn)
```

#### 2.5下拉列表和图片设置  
用于选择划分特征的算法，这里可选ID3或C4.5。  
这里没去研究如何将matplotlib内嵌到wx里了，因此将matplotlib输出图保存，然后直接替换。


```python
    ####下拉列表设置####
    self.choice = wx.Choice(panel,choices = ['ID3','C4.5'],pos=(260,50),size=(50,30))
    self.choice.Bind(wx.EVT_CHOICE,self.SelectSplit)

    ####图片设置####
    DTimage = wx.Image('chushi.jpg',wx.BITMAP_TYPE_JPEG).Rescale(720, 360).ConvertToBitmap() 
    self.bmp = wx.StaticBitmap(panel, -1, DTimage,pos=(30,100)) #转化为wx.StaticBitmap()形式

    self.Centre() 
    self.Show() 
    self.Fit()        
```

### 3.获取数据文件路径事件


```python
####获取数据文件路径事件响应####
def GetFilePath(self,event):
    self.filepath = event.GetString()
```

### 4.打开数据文件路径事件 
这两个可以互补，既可以输入路径打开文件，也可以直接通过对话框选择。


```python
####打开数据文件路径事件响应####
def OpenFileDialog(self,event):        
    wildcard = "csv Files (*.csv)|*.csv" 
    dlg = wx.FileDialog(self, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN)		
    if dlg.ShowModal() == wx.ID_OK: 
        self.file.SetLabel(dlg.GetPath())
    dlg.Destroy() 
```

### 5.加载数据事件  
可能会出现在加载数据时并没有有效路径的异常，此时会提示选择或输入文件路径。


```python
####加载数据事件响应####    
def LoadData(self,event):
    btn = event.GetEventObject().GetLabel() 
    if btn == "LoadData":
        try:
            self.data = pd.read_csv(self.filepath)
            self.data_feature=self.data.columns.values.tolist()[:len(self.data.columns)-1]
            self.featabel = self.data_feature.copy()
            print("数据导入成功")
            self.hint1.SetLabel("数据导入成功!")
            self.hint1.SetForegroundColour((0,255,0))
        except AttributeError:
            print("未获取文件路径")
            self.hint1.SetLabel("请选择或输入文件路径！")
            self.hint1.SetForegroundColour((255,0,0))
    else:
        print("请重新导入")
        self.hint1.SetLabel("数据导入失败，请重新导入!")
        self.hint1.SetForegroundColour((255,0,0))
```

### 6.选择划分算法事件  
选择ID3或C4.5，初始化决策树模型


```python
####选择划分算法事件响应####
def SelectSplit(self, event):
    name = event.GetString()
    #btn = event.GetEventObject().GetLabel()
    if name == 'ID3':
        print("选择划分算法：",name)
        self.hint1.SetLabel("选择了ID3算法")
        self.hint1.SetForegroundColour((0,255,0))
        self.DT_model = DecisionTREE.DecisionTree(split='ID3')


    if name == 'C4.5':
        print("选择划分算法：",name)
        self.hint1.SetLabel("选择了C4.5算法")
        self.hint1.SetForegroundColour((0,255,0))
        self.DT_model = DecisionTREE.DecisionTree(split='C45') 
```

### 7.训练事件  
训练事件可能会出现未载入数据或是未选择划分算法的异常，提示导入数据或选择算法。  
**注意：**每次训练前都需要重新导入数据，否则也会出现异常。这个应该是我的代码写的一些瑕疵，后续会去解决。


```python
####训练事件响应####
def Train(self,event):
    try:
        self.hint1.SetLabel("训练ing")
        self.hint1.SetForegroundColour((0,255,255))
        self.data_model = self.DT_model.create_tree(self.data,self.data_feature)
        self.hint1.SetLabel("训练成功")
        self.hint1.SetForegroundColour((0,255,0))
    except AssertionError as bb:
        print(bb)
        print("请重新导入数据")
        self.hint1.SetLabel("数据已失效，请重新导入数据!")
        self.hint1.SetForegroundColour((255,0,0))
    except AttributeError as aa:
        print(aa)
        print("未导入数据或未选择算法")
        self.hint1.SetLabel("未导入数据或未选择算法，请导入数据或选择算法!")
        self.hint1.SetForegroundColour((255,0,0))
```

### 8.获取测试样本(单个)事件  
测试样本每个特征间以空格或:分割


```python
####获取测试样本事件响应####
def GetSamplVector(self,event):
    sampletext = event.GetString()
    self.SamplVector = re.split(r':| ',sampletext)#正则化划分输入的字符串
    print(self.SamplVector)
```

### 9.预测事件  
可能会有未导入数据或未选择算法的异常，也可能有输入样本数据出错的异常(好像还有IndexError异常，后续补上)


```python
####预测事件响应####
def GetPrediction(self,event):
    try:
        result = self.DT_model.classify(self.data_model,self.featabel,self.SamplVector)
        self.hint2.SetLabel("预测结果："+result)
        self.hint2.SetForegroundColour((255,0,255))
    except AttributeError:
        print("未导入数据或未选择算法")
        self.hint1.SetLabel("未导入数据或未选择算法，请导入数据或选择算法!")
        self.hint1.SetForegroundColour((255,0,0))
    except TypeError as error:
        print(error)
        print("输入的样本数据有误")
```

### 10.绘制决策树事件


```python
####绘制决策树事件响应####
def Plot(self,event):
    try:
        self.hint1.SetLabel("绘图ing")
        self.hint1.SetForegroundColour((0,255,255))
        DDT = DrawDecisionTREE.DrawDecisionTree()
        DDT.createPlot(self.data_model)
        DTimage = wx.Image('dt.jpg', wx.BITMAP_TYPE_JPEG).Rescale(720, 360)
        self.bmp.SetBitmap(wx.BitmapFromImage(DTimage))
        self.hint1.SetLabel("绘图成功")
        self.hint1.SetForegroundColour((0,0,255))
    except AttributeError:
        print("模型不存在，请先训练模型")
        self.hint1.SetLabel("模型不存在，请先训练模型")
        self.hint1.SetForegroundColour((255,0,0))
```

## 四、打包  
用的是Pyinstaller进行打包，chushi.jpg需在打包后，自行拷贝至exe文件同级目录下。  
>\>pyinstaller DTGUI.py -p DecisionTREE.py -p DrawDecisionTREE.py  

打包时出现了这个问题:**This application failed to start because it could not find or load the Qt platform plugin "windows"
in "".**  
[问题解决参考](https://www.cnblogs.com/BlogOfMr-Leo/p/8552385.html)  
后来又出现了上面说了的sip问题。  
大功告成！就是太丑了T T，有时间去学学界面如何美化。
![这里写图片描述](https://img-blog.csdn.net/20180830230221166?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM1OTc5MzE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180830230241296?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTM1OTc5MzE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
