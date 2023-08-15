import asyncio
import codecs
from websockets.server import serve

PORT = 8765
HOST = '127.0.0.1'

async def echo(websocket):
    async for message in websocket:
        print(f'Received: {len(message)} bytes from client {websocket.remote_address}')
        await websocket.send(message)

async def main():
    async with serve(echo, HOST, PORT):
        print(f'Serving on ws://{HOST}:{PORT}...')
        await asyncio.Future()  # run forever

asyncio.run(main())