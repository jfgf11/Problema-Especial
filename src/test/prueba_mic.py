#!/usr/bin/env python3
import pyaudio
from numpy import short,fromstring
from time import sleep

def find_input_device():
    print("----------------------record device list---------------------")
    info = pa.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", pa.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = int(input())
    print("recording via index " + str(index))
    return index

NUM_SAMPLES = 2048
SAMPLING_RATE = 22050
pa = pyaudio.PyAudio()
index = find_input_device()
stream = pa.open(format=pyaudio.paInt16,
                input_device_index=index,
                channels=1, rate=SAMPLING_RATE,
                input=True,
                frames_per_buffer=NUM_SAMPLES)

while (True):
    while stream.get_read_available() < NUM_SAMPLES:
        sleep(0.02)
        audio_data = fromstring(stream.read(stream.get_read_available()), dtype=short)[-NUM_SAMPLES:]
        # Each data point is a signed 16 bit number, so we can normalize by dividing 32*1024
        normalized_data = audio_data / 32768.0
        #intensity = abs(fft(normalized_data))[:NUM_SAMPLES / 2]
        #frequencies = linspace(0.0, float(SAMPLING_RATE) / 2, num=NUM_SAMPLES / 2)
        print('data: ', normalized_data)