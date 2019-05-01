import os
import socketio
import time

from agent import Agent

sio = socketio.Client()
agents = {}

@sio.on('connect')
def on_connect():
    print('SocketIO connected: AI is up. Identifying as the AI')
    sio.emit('connectAI', {'passwd': 'Kore wa watashi no passwd'})

@sio.on('create_learning_agent')
def on_create_learning_agent(data):
    agents[data["id"]] = Agent(sio, data["id"])

@sio.on('action')
def on_action(data):
    agents[data["agentId"]].action(data["data"])

@sio.on('terminate_agent')
def on_terminate_agent(data):
    del agents[data["agentId"]]

@sio.on('disconnect')
def on_disconnect():
    print("Socket disconnected. Attempting to reconnect")
    connect()

address = "http://localhost:8080"
if hasattr(os.environ, "ENV"):
    print("Variable ENV is present")
    if os.environ["ENV"] == "heroku":
        print("variable ENV is heroku")
        address = "http://blobwar.herokuapp.com/"

print("Address is: " + address)

def connect():
    try:
        sio.connect(address)
    except:
        print("Socket connection failed. Retrying in 6 seconds")
        time.sleep(6)
        connect()

connect()
