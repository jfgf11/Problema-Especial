#!/usr/bin/env python3
import pyaudio
import struct
import numpy as np
import threading
import time
import librosa
from tensorflow.keras.models import model_from_json
from termcolor import colored

modelo = 'modelo'
pesos = 'pesos_modelo'

rutaModelo = "./src/modelo/" + modelo + ".json"
rutaPesos = "./src/modelo/" + pesos + ".h5"

SAMPLE_RATE = 22050
window_length_stft_mfcc = 0.032
window_length_stft_esp = 0.025
Step_size_stft = 0.025
ventana_Tiempo_ = 0.100
INPUT_FRAMES_PER_BLOCK = int(SAMPLE_RATE * ventana_Tiempo_)


def get_rms(block):
    return np.sqrt(np.mean(np.square(block)))


def cargarModelo(pRutaModelo, pRutaPesos):
    archivo_json = open(pRutaModelo, 'r')
    modelo_json = archivo_json.read()
    archivo_json.close()
    m = model_from_json(modelo_json)
    m.load_weights(pRutaPesos)
    return m


class AudioHandler(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.model = self.iniciar()
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

    def open_mic_stream(self):
        device_index = self.find_input_device()
        stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
                              input_device_index=device_index, frames_per_buffer=INPUT_FRAMES_PER_BLOCK)
        return stream

    def iniciar(self):
        print('Inicio')
        self.model = cargarModelo(rutaModelo, rutaPesos)
        return self.model

    def processBlock(self, p_audio):
        audio_n = p_audio / 1.0

        MFCC = librosa.feature.mfcc(y=audio_n, sr=SAMPLE_RATE, n_mfcc=20,
                                    n_fft=int(window_length_stft_mfcc * SAMPLE_RATE),
                                    hop_length=int(Step_size_stft * SAMPLE_RATE), htk=True)
        esp = librosa.feature.melspectrogram(y=audio_n, sr=SAMPLE_RATE, n_fft=int(window_length_stft_esp * SAMPLE_RATE),
                                             hop_length=int(Step_size_stft * SAMPLE_RATE))
        alto, ancho = MFCC.shape
        MFCC = np.reshape(MFCC, (-1, alto, ancho, 1), 'F')
        alto, ancho = esp.shape
        esp = np.reshape(esp, (-1, alto, ancho, 1), 'F')
        clase = np.argmax(self.model.predict([esp, MFCC]), axis=-1)

        if clase[0] == 0:
            print(colored('0', 'magenta'))
        elif clase[0] == 1:
            print(colored('1', 'green'))
        elif clase[0] == 2:
            print(colored('2', 'blue'))
        elif clase[0] == 3:
            print(colored('3', 'red'))

    def listen(self):
        try:
            raw_block = self.stream.read(INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False)
            count = len(raw_block) / 2
            formato = '%dh' % count
            muestra = np.array(struct.unpack(formato, raw_block))
        except Exception as e:
            print('Error recording: {}'.format(e))
            return

        return muestra


if __name__ == '__main__':
    audio = AudioHandler()
    run = True
    while run:
        # print(threading.active_count())
        toProcess =audio.listen()
        audio.processBlock(toProcess)
        # x = threading.Thread(target=self.processBlock, args=(toProcess,))
        # x.start()
        if threading.active_count() > 10:
            run = False

    while threading.active_count() > 1:
        print('Cerrando Threads')
        time.sleep(0.5)

    print('Se han cerrado todos los threads')
