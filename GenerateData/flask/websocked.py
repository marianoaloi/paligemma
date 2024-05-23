import asyncio
import json
import sys
from typing import List
import websockets

from entity import ItemList



class WebSocketPaligemma:
    CONNECTIONS = set()
    def __init__(self,images:dict={}) -> None:
        
        self.images:dict=images
    
    def write(self,websocket, event):
        image=event["image"]
        text=event["text"]
        
    
    async def register(self,websocket):
        self.CONNECTIONS.add(websocket)
        try:
            async for message in websocket:
                event = json.loads(message)
                action=event["action"]
                if action == "getAll":
                    await websocket.send(json.dumps(self.images))
                elif action == "countColaborators":
                    await websocket.send(str(len(self.CONNECTIONS)))
                elif action == "write":
                    await self.write(websocket, event)
        except Exception as e:
            await websocket.send(str(e))
        finally:
            self.CONNECTIONS.remove(websocket)    
    


    async def main(self):
        async with websockets.serve(self.register, "0.0.0.0", 8765) as w:
            print([ws.getsockname() for ws in w.sockets])
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(WebSocketPaligemma().main())