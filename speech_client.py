#!/usr/bin/env python

import pyaudio
import socket
import struct
from smart_asistant_server import types, HOST, PORT


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("*recording")

frames = []

type_value = types["speech"]

packer = struct.Struct('I')
packed_data = packer.pack(type_value)
s.send(packed_data)

for i in range(0, int(RATE/CHUNK*RECORD_SECONDS)):
    data  = stream.read(CHUNK)
    frames.append(data)
    s.sendall(data)

print("*done recording")

stream.stop_stream()
stream.close()
p.terminate()
s.close()

print("*closed")