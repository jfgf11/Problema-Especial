#!/usr/bin/env python3
import pyaudio
import struct
import math
import numpy as np
#from scipy import signal
import matplotlib.pyplot as plt
import threading
import time
import librosa


RATE = 22000
INPUT_BLOCK_TIME = 0.032 # 32 ms
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)

def get_rms(block):
    return np.sqrt(np.mean(np.square(block)))

class AudioHandler(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.plot_counter = 0

    def stop(self):
        self.stream.close()

    def find_input_device(self):
        print("----------------------record device list---------------------")
        info = self.pa.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", self.pa.get_device_info_by_host_api_device_index(0, i).get('name'))

        print("-------------------------------------------------------------")

        index = int(input())
        print("recording via index " + str(index))

        return index

    def open_mic_stream( self ):
        device_index = self.find_input_device()

        stream = self.pa.open(  format = pyaudio.paInt16,
                                channels = 1,
                                rate = RATE,
                                input = True,
                                input_device_index = device_index,
                                frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

        return stream

    def processBlock(self, audio):
        print ("Processing started")
        start = time.time()
        #f, t, Sxx = signal.spectrogram(audio, RATE)

        audio = audio / 1.0
        ps=librosa.feature.melspectrogram(audio,RATE)

        end = time.time()
        print("Processing finished")
        print(end - start)
        if (end-start>0.032):
            print ("Tiempo Superado")
        return


    def listen(self):
        try:
            raw_block = self.stream.read(INPUT_FRAMES_PER_BLOCK, exception_on_overflow = False)
            count = len(raw_block) / 2
            format = '%dh' % (count)
            audio = np.array(struct.unpack(format, raw_block))
        except Exception as e:
            print('Error recording: {}'.format(e))
            return

        #amplitude = get_rms(audio)
        toProcess=audio
        x = threading.Thread(target=self.processBlock, args=(toProcess,))
        x.start()



if __name__ == '__main__':
    audio = AudioHandler()
    run=True
    while (run):
        print(threading.active_count())
        audio.listen()
        if threading.active_count()>10:
            run=False

    while threading.active_count()>1:
        print('Cerrando Threads')
        time.sleep(0.5)

    print('Se han cerrado todos los threads')