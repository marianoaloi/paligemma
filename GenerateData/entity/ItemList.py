
import hashlib
import os
import traceback
from PyQt5.QtGui import QImage,QImageReader

SizeMaxMozaic=300

FILE_DEFAULT="data.jsonl"
class ItemList:
    
    
    
    def __init__(self,dir:str=None,fileName:str=None,description:str="",question:str = "") -> None:
        self.pathPhoto:str=os.path.join(dir,fileName)
        self.fileName:str=fileName
        self.description:str=description
        self.question = question
        self.__id:str=hashlib.md5(self.pathPhoto.encode("utf-8")).hexdigest()
        
    def populateDigitalImage(self):
        self.image = None
        try:
            self.image = self.getFileSystemImage(self.pathPhoto)
        except Exception as e:
                traceback.print_exception(e)
                
    def getFileSystemImage(self,path)->QImage:

        
        reader = QImageReader(path)
        
        # Check if the image format is supported
        if not reader.canRead():
            raise RuntimeError(f"Error: Unsupported image format for '{path}'.")

        # Read the image
        imageQ = reader.read()

        # Check for errors during reading
        if imageQ.isNull():
            error_str = reader.errorString()
            raise RuntimeError(f"Error reading image: {error_str}")
        imageQ=imageQ.scaled(SizeMaxMozaic,SizeMaxMozaic)
        return imageQ