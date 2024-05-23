import asyncio
import getopt
import json
import mimetypes
import os
import shlex
import subprocess
import sys
import time
import traceback

from PIL import Image,ImageFile
import numpy as np

from flask.websocked import WebSocketPaligemma
os.environ['QT_IMAGEIO_MAXALLOC'] = "0"
ImageFile.LOAD_TRUNCATED_IMAGES = True


from typing import List
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


from PyQt5 import sip , uic

from mediamaloipackage.splashscreen import SplashScreen
from mediamaloipackage.HelpComponent import HelpComponent

from mediamaloipackage.ValidateFile import ValidateFile

from entity.ItemList import ItemList,SizeMaxMozaic

FILE_DEFAULT="data.jsonl"

class PlainTextEditDelegate(QStyledItemDelegate):
    def __init__(self, parent=None,saveMethod=None):
        super().__init__(parent)
        self.saveFile=saveMethod

    def createEditor(self, parent, option, index):
        """Creates a QPlainTextEdit widget for cell editing."""
        editor = QPlainTextEdit(index.data(),parent)
        editor.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        editor.setStyleSheet("background-color:pink")
        editor.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        if(self.saveFile):
            editor.leaveEvent = self.saveFile
            editor.focusOutEvent = (self.saveFile)
        # Optionally, customize the editor (e.g., font, size)
        return editor

    
class HtmlDelegate(QStyledItemDelegate):
    def __init__(self, parent: QObject | None = ...) -> None:
        super().__init__(parent)
    def paint(self, painter, option, index):
        item = index.model().data(index, Qt.DisplayRole)
        label = QLabel(item)
        # label.setMaximumWidth(SizeMaxMozaic)
        label.setMinimumWidth(SizeMaxMozaic)
        label.setMaximumHeight(SizeMaxMozaic)
        label.setMinimumHeight(int(SizeMaxMozaic/2))
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)  # Enable HTML rendering
        # label.adjustSize()
        painter.save()
        painter.translate(option.rect.topLeft())
        label.render(painter)
        painter.restore()   



class GeneratePaligemmaData(QMainWindow):
    def __init__(self,**kwargs):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),"main.ui"), self)
        
        self.helpDialog = HelpComponent()
        self.helpDialog.populateShortCut(QKeySequence(Qt.Key.Key_F1), self,self.helpDialog.createHelp,"Help")
        
        
        # self.tableData.cellChanged.connect(self.saveFile)
        
        self.directory = None
        self.images:dict={}
        
        
        self.actionStart_WebSocket   .triggered.connect(lambda : asyncio.run(self.websocket()))
        
    def init(self,progressBarSplash):
        def progressBar(v):
            if(progressBarSplash):
                progressBarSplash(v);
                t = time.time()
                while time.time() < t + 0.1:
                    QApplication.instance().processEvents()
        progressBar(42)


        try:
            opts, args = getopt.getopt(sys.argv[1:], "h:f:i:p")


            def getParam(paramm)->str:
                lparm=[x[1] for x in opts if x[0] == paramm]
                return lparm[0] if len(lparm) > 0 else None

            self.directory = getParam("-f")
        except getopt.GetoptError:
            print("ChoiceVideos -f <file>")


        progressBar(1)
        
        if(self.directory):
            self.openDirectory(progressBar)
                    
    async def websocket(self)    :
        
        WebSocketPaligemma(self.images).main()
    def openDirectory(self,progressBar=None):
        self.images={}
        
        self.show()
                    
        self.fillTable(progressBar)
    def setCounts(self,field:QLabel,label:str,count:int):
        field.setText(f"{label}{str(count)}")
         
    def getImagesYeild(self,progressBar=None):
        paths=set()
        self.total=0
        filled=0
        empty=0
        if(os.path.exists(os.path.join(self.directory,FILE_DEFAULT))):
            fileNamePath=(os.path.join(self.directory,FILE_DEFAULT))
            
                    
            with open(fileNamePath) as jsonl:
                lines=jsonl.readlines()
                if(lines):
                    bias=100/len(lines)
                    for index, line in enumerate(lines):
                        obj=json.loads(line)
                        paths.add(obj["image"])
                        image = ItemList(self.directory,obj["image"],obj["suffix"])
                        self.images[image.pathPhoto]=image
                        yield image
                        progressBar(int(index*bias))
                        self.total+=1
                        filled+=1
                        self.setCounts(self.totalCount,"Total:",self.total)
                        self.setCounts(self.filledCount,"Filled:",filled)
        
        lines=[file for file in 
                [os.path.join(root,file) 
                for root,_,files in os.walk(self.directory) 
                for file in files if not file in paths ] 
                if(ValidateFile.validateFileIMG(file))]
        if(lines):
            bias=100/len(lines)
            for index, line in enumerate(lines):
                
                    image = ItemList(os.path.dirname(line),os.path.basename(line),"")
                    self.images[image.pathPhoto]=image
                    yield image
                    progressBar(int(index*bias))
                    self.total+=1
                    empty+=1
                    self.setCounts(self.totalCount,"Total:",self.total)
                    self.setCounts(self.emptyCount,"Empty:",empty)
        
    def fillTable(self,progressBar=None):
        table=self.tableData
        
        table.clear()
        table.clearContents()
        table.setColumnCount(2)
        # table.setRowCount(len(self.images))
        table.columnWidth(SizeMaxMozaic +10)
        table.rowHeight(SizeMaxMozaic +10)
        imageSizeWithMargins=(SizeMaxMozaic + 10 )
        
        for column in (range(table.columnCount())):
            table.setColumnWidth(column, imageSizeWithMargins)
        
        try:
            
                table.setItemDelegateForColumn(1, 
                    PlainTextEditDelegate(table,saveMethod=self.saveFileEvent)) 
        except Exception as e:
            traceback.print_exception(e)
            
            
        for row, data in enumerate(self.getImagesYeild(progressBar)):
            
            if(not data.image):
                continue
            
            table.insertRow(row) 
            
            item = QTableWidgetItem()
            item.setSizeHint(QSize((SizeMaxMozaic + 10), SizeMaxMozaic + 10))
            item.setData(Qt.DecorationRole, data.image)        
            # item.widget = image
            table.setItem(row, 0, item)
            
            item = QTableWidgetItem(data.description)
            # item.setWordWrap(True)
            # textPlain=QPlainTextEdit("CACACACACACACACACA",table)
            # textPlain.enterEvent = self.saveFileEvent
            # textPlain.leaveEvent = self.saveFileEvent           
            # textPlain.focusOutEvent = (self.saveFileEvent)
            # textPlain.setStyleSheet("background-color:red")
            
            # textPlain.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
            # textPlain.setMinimumSize(300, 300)
            # item.setData(Qt.DecorationRole,textPlain)
            setattr(item,"path",data.fileName)
            
            item.setSizeHint(QSize((SizeMaxMozaic + 10), SizeMaxMozaic + 10))
            table.setItem(row, 1, item)
            
            table.setRowHeight(row, imageSizeWithMargins)
        table.columnResized(SizeMaxMozaic+10,0,SizeMaxMozaic+10)
        
        
        self.zoomAll  ()
        
    


    def saveFileEvent(self,event=None)  :
        self.saveFile()
    def saveFile(self):
        tab=self.tableData
        with open(os.path.join(self.directory,FILE_DEFAULT),'w') as fjsonl:
            items=[text for text,row in [
              (tab.item(row,1),row) for row in range(tab.rowCount())
            ] if text and text.text()]
            filled=len(items)
            self.setCounts(self.filledCount,"Filled:",filled)
            self.setCounts(self.emptyCount,"Empty:",self.total - filled)
            for line in items:
                fjsonl.write(json.dumps({"prefix": "", "suffix": line.text(),"image": line.path}))
                fjsonl.write("\n")
                


        
    def zoomAll(self)   :
            
        imageSizeWithMargins=(SizeMaxMozaic + 10 )
        self.tableData.setMinimumWidth(imageSizeWithMargins * self.tableData.columnCount())
        self.tableData.setMinimumHeight(imageSizeWithMargins * self.tableData.rowCount())
        self.tableData.columnWidth(imageSizeWithMargins)
        self.tableData.rowHeight(imageSizeWithMargins)
        
        for row in (range(self.tableData.rowCount())):
            self.tableData.setRowHeight(row, imageSizeWithMargins)  # Set height for third row
        for column in (range(self.tableData.columnCount())):
            self.tableData.setColumnWidth(column, imageSizeWithMargins)  # Set width for first column
            
            
if __name__ == "__main__":
    app = QApplication([])
    app.setWindowIcon(
        QIcon(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                    'images/icon.jpeg'
                )
            )
        )
    
    splash = SplashScreen(img=os.path.join(os.path.dirname(os.path.abspath(__file__)),"splashscreen.svg"))
    splash.show()
    window = GeneratePaligemmaData()
    window.show()
    window.init(splash.progressBar)
    splash.finish(window)
    sys.exit(app.exec())    