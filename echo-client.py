
import socket
import stt 

HOST = '192.168.0.2'

PORT = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

client_socket.sendall(stt.question.encode()) ## input to server
data = client_socket.recv(1024)

print('Received text', repr(data.decode())) ## received from server




client_socket.close()
