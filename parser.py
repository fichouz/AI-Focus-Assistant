import struct
import numpy as np

class Paket:
    def __init__(self, id, ts, data):
        self.id = id
        self.ts = ts
        self.data = data

def crc16_update(crc, data):
    crc ^= data
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc >>= 1
    return crc


def crc16_compute(data):
    crc = 0xFFFF
    for b in data:
        crc = crc16_update(crc, b)
    return crc
