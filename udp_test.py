import socket
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('localhost', 12348))
sock.setblocking(False)  # or sock.settimeout(0.1)

while True:
    try:
        data, addr = sock.recvfrom(1024)
        values = struct.unpack('22d', data)
        print(values)
    except BlockingIOError:
        pass  # no data yet, continue
