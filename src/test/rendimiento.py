#!/usr/bin/env python3
import pyaudio
import struct
import numpy as np
import time
import librosa
from tensorflow.keras.models import model_from_json
import psutil

modelo = 'modelo_final'
pesos = 'pesos_modelo_final'

rutaModelo = "./src/modelo/" + modelo + ".json"
rutaPesos = "./src/modelo/" + pesos + ".h5"

SAMPLE_RATE = 22050
window_length_stft_mfcc = 0.032
window_length_stft_esp = 0.025
Step_size_stft = 0.01
ventana_Tiempo_ = 0.450
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
        self.mfcc = None
        self.esp = None
        self.modelo = None
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.iniciar()
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
        self.modelo = cargarModelo(rutaModelo, rutaPesos)

    def preprocesing(self, p_audio):

        audio_n = p_audio / 1.0

        MFCC = librosa.feature.mfcc(y=audio_n, sr=SAMPLE_RATE, n_mfcc=20,
                                    n_fft=int(window_length_stft_mfcc * SAMPLE_RATE),
                                    hop_length=int(Step_size_stft * SAMPLE_RATE), htk=True)
        esp = librosa.feature.melspectrogram(y=audio_n, sr=SAMPLE_RATE, n_fft=int(window_length_stft_esp * SAMPLE_RATE),
                                             hop_length=int(Step_size_stft * SAMPLE_RATE))
        alto, ancho = MFCC.shape
        self.mfcc = np.reshape(MFCC, (-1, alto, ancho, 1), 'F')
        alto, ancho = esp.shape
        self.esp = np.reshape(esp, (-1, alto, ancho, 1), 'F')

    def predict(self):
        np.argmax(self.modelo.predict([self.esp, self.mfcc]), axis=-1)

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

    print('--------------------- Iniciando Pruebas -------------------------')

    ram_inicial = psutil.virtual_memory().available
    audio = AudioHandler()
    ram_operacion = psutil.virtual_memory().available

    tiempo_recoleccion = []
    memoria_audio = []
    memoria_esp_mfcc = []
    tiempo_preprocesamiento = []
    tiempo_prediccion = []

    for k in range(100):
        m1 = psutil.virtual_memory().available
        start = time.time()
        toProcess = audio.listen()
        end = time.time()
        m2 = psutil.virtual_memory().available  # calcula peso del audio
        memoria_audio.append(int(m2) - int(m1))
        tiempo_recoleccion.append(float(end) - float(start))

        m1 = psutil.virtual_memory().available
        start = time.time()
        audio.preprocesing(toProcess)
        end = time.time()  # calcula tiempo de pre-procesamiento
        m2 = psutil.virtual_memory().available  # calcula peso del espectograma y mfcc
        memoria_esp_mfcc.append(int(m2) - int(m1))
        tiempo_preprocesamiento.append(float(end) - float(start))

        start = time.time()
        audio.predict()
        end = time.time()  # calcula tiempo de prediccion
        tiempo_prediccion.append(float(end) - float(start))

    print('--------------------- Memoria -------------------------')
    print('memoria ocupada por el modelo ', (int(ram_inicial) - int(ram_operacion)) / 1000000, 'MB')
    print('memoria inicial ', int(ram_inicial) / 1000000, 'MB')

    print('--------------------- Tiempo de recoleccion -------------------------')
    print('media ', np.mean(tiempo_recoleccion) * 1000, 'ms')
    print('mediana ', np.median(tiempo_recoleccion) * 1000, 'ms')

    print('--------------------- Tiempo de procesamiento -------------------------')
    print('media ', np.mean(tiempo_preprocesamiento) * 1000, 'ms')
    print('mediana ', np.median(tiempo_preprocesamiento) * 1000, 'ms')

    print('--------------------- Tiempo de prediccion -------------------------')
    print('media ', np.mean(tiempo_prediccion) * 1000, 'ms')
    print('mediana ', np.median(tiempo_prediccion) * 1000, 'ms')

    print('--------------------- Memoria ocupada audio -------------------------')
    print('media ', np.mean(memoria_audio))
    print('mediana ', np.median(memoria_audio))

    print('------------------- Memoria ocupada esp y mfcc -----------------------')
    print('media ', np.mean(memoria_esp_mfcc))
    print('mediana ', np.median(memoria_esp_mfcc))
