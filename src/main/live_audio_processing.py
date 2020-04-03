#!/usr/bin/env python3
import pyaudio
import struct
import numpy as np
#from scipy import signal
import threading
import time
import librosa
from tensorflow.keras.models import model_from_json


rutaModelo=".Modelo_Male_200 (1).json"
rutaPesos=".Pesos_Modelo_Male_200 (1).h5"
SAMPLE_RATE=22050
window_length_stft = 0.025
Step_size_stft = 0.010
ventana_Tiempo_ = 0.050
INPUT_FRAMES_PER_BLOCK = int(SAMPLE_RATE * ventana_Tiempo_)
modelo1=None

def get_rms(block):
    return np.sqrt(np.mean(np.square(block)))

class AudioHandler(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.modelo1=self.iniciar()
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
                                rate = SAMPLE_RATE,
                                input = True,
                                input_device_index = device_index,
                                frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

        return stream

    def cargarModelo(self,pRutaModelo, pRutaPesos):
        archivo_json = open(pRutaModelo, 'r')
        modelo_json = archivo_json.read()
        archivo_json.close()
        modelo = model_from_json(modelo_json)
        modelo.load_weights(pRutaPesos)
        return modelo

    def iniciar(self):
        self.modelo1 = self.cargarModelo(rutaModelo, rutaPesos)
        self.modelo1.compile(loss='sparse_categorical_crossentropy', optimizer="rmsprop",
                        metrics=['sparse_categorical_accuracy'])

    def processBlock(self, audio):
        #print ("Processing started")
        start = time.time()
        #f, t, Sxx = signal.spectrogram(audio, RATE)

        audio = audio / 1.0
        #ps=librosa.feature.melspectrogram(audio,RATE)

        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE,
                                    n_mfcc=13)  # ,n_fft = int(window_length_stft*SAMPLE_RATE), hop_length = int(Step_size_stft*SAMPLE_RATE), htk=True )

        print(np.argmax(self.modelo1.predict(mfcc), axis=-1))

        end = time.time()
        #print("Processing finished")
        #print(end - start)
        if (end-start>ventana_Tiempo_):
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