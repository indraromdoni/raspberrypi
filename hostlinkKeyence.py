import socket

# Create a client socket
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);

# Connect to the server
clientSocket.connect(("196.62.10.100",8501)); 

# Send data to server
data = "WR MR0 0\r"
clientSocket.send(data.encode());
decodedata = data.encode()
print(decodedata.decode())
 

# Receive data from server

dataFromServer = clientSocket.recv(1024);

# Print to the console
print(dataFromServer.decode());

dataread = "RD MR0\r"
clientSocket.send(dataread.encode())
msg = clientSocket.recv(1024)
print(msg.decode())