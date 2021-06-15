
import socket
import stt
import os

HOST = '192.168.0.5'

PORT = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

client_socket.sendall(stt.question.encode()) ## input to server
data = client_socket.recv(1024)
result = repr(data.decode())
print('Received text', result) ## received from server

client_socket.close()

