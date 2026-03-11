"""
Simple UDP listener on port 9 to test if WoL magic packets arrive.
Run as admin if needed (binding to low ports on Windows may require it).
"""
import socket

PORT = 9
MAGIC_PREFIX = b'\xff' * 6

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', PORT))
print(f"Listening for UDP packets on port {PORT}...")
print("Send a WoL packet from your phone, then check here.\n")

while True:
    data, addr = sock.recvfrom(1024)
    print(f"Received {len(data)} bytes from {addr}")
    print(f"  Hex: {data.hex()}")
    if data[:6] == MAGIC_PREFIX:
        mac = ':'.join(f'{b:02x}' for b in data[6:12])
        print(f"  -> Magic packet detected! Target MAC: {mac}")
    else:
        print(f"  -> Not a magic packet")
    print()
