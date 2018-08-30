# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:16:44 2018

@author: htshinichi
"""
from PyQt5 import sip
import re
import os
import wx
import pandas as pd
import DecisionTREE
import DrawDecisionTREE 
class Mywin(wx.Frame):
    def __init__(self, parent, title):
        super(Mywin, self).__init__(parent, title = title,size = (800,600))  
        panel = wx.Panel(self)
        ####字体设置####
        font_title = wx.Font(30, wx.ROMAN, wx.NORMAL, wx.BOLD)#设置字体(大小、样式等)      
        font_btn = wx.Font(15, wx.ROMAN, wx.ITALIC, wx.NORMAL)
        font_text = wx.Font(15,wx.ROMAN,wx.NORMAL,wx.NORMAL)
        font_hint1 = wx.Font(10,wx.ROMAN,wx.NORMAL,wx.NORMAL)
        font_hint2 = wx.Font(15,wx.ROMAN,wx.NORMAL,wx.NORMAL)
        ####固定文本设置####      
        self.title = wx.StaticText(panel,-1,"决策树",pos=(650,0),size=(80,30),style=wx.ALIGN_RIGHT)#pos用于设置位置、size用于设置大小
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
        
        ####输入文本设置####
        self.file = wx.TextCtrl(panel,-1, pos=(50,10),size=(200,20))
        self.file.Bind(wx.EVT_TEXT,self.GetFilePath)
        
        self.sample = wx.TextCtrl(panel,-1, pos=(200,480),size=(200,50))
        self.sample.Bind(wx.EVT_TEXT,self.GetSamplVector)
        
        ####按键设置####
        self.btn = wx.Button(panel,-1,"Select",pos=(260,10),size=(80,20)) 
        self.btn.Bind(wx.EVT_BUTTON,self.OpenFileDialog)
        self.btn.SetFont(font_btn)
        
        self.btn = wx.Button(panel,-1,"LoadData",pos=(350,10),size=(90,20)) 
        self.btn.Bind(wx.EVT_BUTTON,self.LoadData)
        self.btn.SetFont(font_btn)
        
        self.choice = wx.Choice(panel,choices = ['ID3','C4.5'],pos=(260,50),size=(50,30))
        self.choice.Bind(wx.EVT_CHOICE,self.SelectSplit)
        
     
        self.btn = wx.Button(panel,-1,"Train",pos=(330,45),size=(50,30)) 
        self.btn.Bind(wx.EVT_BUTTON,self.Train)
        self.btn.SetFont(font_btn)  
        
        self.btn = wx.Button(panel,-1,"PlotDT",pos=(50,480),size=(80,50)) 
        self.btn.Bind(wx.EVT_BUTTON,self.Plot)
        self.btn.SetFont(font_btn) 
        
        self.btn = wx.Button(panel,-1,"Predict",pos=(420,480),size=(80,50)) 
        self.btn.Bind(wx.EVT_BUTTON,self.GetPrediction)
        self.btn.SetFont(font_btn)
        
        ####图片设置####
        DTimage = wx.Image('chushi.jpg',wx.BITMAP_TYPE_JPEG).Rescale(720, 360).ConvertToBitmap() 
        self.bmp = wx.StaticBitmap(panel, -1, DTimage,pos=(30,100)) #转化为wx.StaticBitmap()形式


        self.Centre() 
        self.Show() 
        self.Fit()        
    
    
    ####3获取数据文件路径事件响应####
    def GetFilePath(self,event):
        self.filepath = event.GetString()
    
    ####4打开数据文件路径事件响应####
    def OpenFileDialog(self,event):        
        wildcard = "csv Files (*.csv)|*.csv" 
        dlg = wx.FileDialog(self, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN)		
        if dlg.ShowModal() == wx.ID_OK: 
            self.file.SetLabel(dlg.GetPath())
        dlg.Destroy() 
                
    ####5加载数据事件响应####    
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
            
    ####6选择划分算法事件响应####
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
            
    ####7训练事件响应####
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
            
    ####8获取测试样本事件响应####                   
    def GetSamplVector(self,event):
        sampletext = event.GetString()
        self.SamplVector = re.split(r':| ',sampletext)
        print(self.SamplVector) 

    ####9预测事件响应####
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
            
    ####10绘制决策树事件响应####
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
      
if __name__ == '__main__':
    try:
        del app
    except NameError:
        print("这个对象不存在,创建一个新的对象")             
    app = wx.App() 
    Mywin(None,  'HuaHuaDT') 
    app.MainLoop()
