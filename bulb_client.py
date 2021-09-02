#!/usr/bin/env python

import socket
import struct
from smart_asistant_server import types, HOST, PORT, BULB_NAME

from gpiozero import LED


led = LED(5)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

type_value = types["register"]

packer = struct.Struct('I')
name_packer = struct.Struct('2s')

packed_data = packer.pack(type_value)
s.send(packed_data)
packed_data = name_packer.pack(BULB_NAME)
s.send(packed_data)

while True:
    data = s.recv(4)
    state = packer.unpack(data)
    if state == 0:
        led.off()
    else:
        led.on()
