import asyncio
import json
import os
import sys
import traceback
from typing import List
import websockets

from entity.ItemList import ItemList, FILE_DEFAULT


from mediamaloipackage.ValidateFile import ValidateFile

from webflask.webpaligemma import WebPaligemma

class WebSocketPaligemma:
    CONNECTIONS = set()
    def __init__(self,images:dict={},directory:str = None) -> None:
        
        self.images:dict=images
        
        self.directory = directory
    
    async def write(self,event,userId):
        
        def broadcast(image :ItemList):
            # delattr(image,"image")
            websockets.broadcast(self.CONNECTIONS, json.dumps({"type":"image","image":image.__dict__,"userId":userId}))
        if(not "image" in event):
            raise Exception("Must put the image value")
        
        if(not "text" in event):
            raise Exception("Must put the text value")
        image=event["image"]
        text=event["text"]
        
        img=self.images.get(image)
        if(not img):
            raise Exception("The server not has this image. restart the service.")
        img.description=text
        broadcast(img)
        with open(os.path.join(self.directory,FILE_DEFAULT),'w') as fjsonl:
            items=[itemImage for itemImage in self.images.values() if itemImage.description]
            for line in items:
                fjsonl.write(json.dumps({"prefix": "", "suffix": line.description,"image": line.fileName}))
                fjsonl.write("\n")
        # await self.resume()
        
    async def  readDirectory(self)->bool:
        def broadcast(image :ItemList):
            # delattr(image,"image")
            websockets.broadcast(self.CONNECTIONS, json.dumps({"type":"image","image":image.__dict__}))
        paths=set()
        if(os.path.exists(os.path.join(self.directory,FILE_DEFAULT))):
            fileNamePath=(os.path.join(self.directory,FILE_DEFAULT))
            
                    
            with open(fileNamePath) as jsonl:
                lines=jsonl.readlines()
                if(lines):                    
                    for line in lines:
                        obj=json.loads(line)
                        if not obj["image"] in paths:
                            paths.add(obj["image"])
                            image = ItemList(self.directory,obj["image"],obj["suffix"])
                            self.images[image.pathPhoto]=image
                            broadcast(image)
        
        lines=[file for file in 
                [os.path.join(root,file) 
                for root,_,files in os.walk(self.directory) 
                for file in files if not file in paths ] 
                if(ValidateFile.validateFileIMG(file))]
        if(lines):            
            for line in lines:               
                    image = ItemList(os.path.dirname(line),os.path.basename(line),"")
                    self.images[image.pathPhoto]=image
                    broadcast(image)
        return True
    async def resume(self):
        total=len(self.images.values())
        filleds=len([itemImage for itemImage in self.images.values() if itemImage.description])
        websockets.broadcast(self.CONNECTIONS, json.dumps({"type":"resume","count":len(self.CONNECTIONS),"total":total,"filled":filleds,"empty":total-filleds}))
    async def register(self,websocket):
        self.CONNECTIONS.add(websocket)
        await websocket.send(json.dumps({"type":"id","id":str(websocket.id)}))
        try:
            async for message in websocket:
                try:
                    event = json.loads(message)
                    if(not "action" in event):
                        raise Exception("Must put the action value")
                    action=event["action"]
                    if action == "getAll":
                        await websocket.send(json.dumps({"type":"images","images":[x.__dict__ for x in self.images.values()]}))
                    elif action == "count":
                        await websocket.send(json.dumps({"type":"count","count":str(len(self.CONNECTIONS))}))
                    elif action == "write":
                        await self.write( event,str(websocket.id))
                    elif action == 'resume':
                        await self.resume()
                    elif action == 'ping':
                        await websocket.send(json.dumps({"type":'pong'}))
                    elif action == 'readdir':
                        if(not "directory" in event):
                            raise Exception("Choice a directory to read")
                        dir=event["directory"]
                        self.directory = dir                    
                        await self.readDirectory()
                    else :
                        raise Exception("No action choiced")
                except Exception as e:
                    await websocket.send(json.dumps({"type":"error","msg":traceback.format_exception (e)}))
        finally:
            self.CONNECTIONS.remove(websocket)    
    


    async def main(self,runForever:bool=True):
        async with websockets.serve(self.register, "0.0.0.0", 8765) as w:
            print([ws.getsockname() for ws in w.sockets])
            if(runForever):
                await asyncio.Future()  # run forever

if __name__ == "__main__":
    web=WebPaligemma()
    asyncio.run(WebSocketPaligemma().main(True))