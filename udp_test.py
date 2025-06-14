import socket
import struct
import time

udp_ip = '127.0.0.1'
udp_recv = 12348
udp_send = 12349
BUFFER_SIZE = 8192

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((udp_ip, udp_recv))

msg_count = 0
start_time = time.time()

try:
    while True:
        data, addr = sock.recvfrom(4096)
        msg_count += 1
        elapsed = time.time() - start_time
        if len(data) >= 18 * 8:  # 18 doubles, 8 bytes each
            values = struct.unpack('18d', data[:18*8])
            print(f"Unpacked values: {values}")
        else:
            print(f"Received data too short to unpack 18 doubles: {len(data)} bytes")
except KeyboardInterrupt:
    print("\nStopped listening.")
finally:
    sock.close()