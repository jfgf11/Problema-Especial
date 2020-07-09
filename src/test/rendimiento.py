#!/usr/bin/env python3
import pyaudio
import struct
import numpy as np
import time
import librosa
from tensorflow.keras.models import model_from_json
import psutil


class AudioHandler(object):

    def __init__(self):
        self.model_source = 'model/final_model.json'
        self.model_weights = 'model/final_model_weights.h5'
        self.sample_rate = 22050
        self.window_length_stft_mfcc = 0.032
        self.window_length_stft_esp = 0.025
        self.step_size_stft = 0.01
        self.time_window = 0.450
        self.input_frames_per_block = int(self.sample_rate * self.time_window)

        self.mfcc = None
        self.esp = None
        self.model = self.load_model()
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()

    def stop(self):
        self.stream.close()

    def find_input_device(self):
        """
        list all the input devices
        :return: index of the selected input device
        """
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
        """
        starts the stream to record audio signals
        :return: stream
        """
        device_index = self.find_input_device()
        stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True,
                              input_device_index=device_index, frames_per_buffer=self.input_frames_per_block)
        return stream

    def load_model(self):
        """
        loads the model and its weights
        :return: pre trained model
        """
        archivo_json = open(self.model_source, 'r')
        modelo_json = archivo_json.read()
        archivo_json.close()
        m = model_from_json(modelo_json)
        m.load_weights(self.model_weights)
        return m

    def preprocesing(self, p_audio):
        """
        gets the mfcc and mel spectrogram from the audio signal
        :param: p_audio: raw audio signal
        """

        audio_n = p_audio / 1.0

        MFCC = librosa.feature.mfcc(y=audio_n, sr=self.sample_rate, n_mfcc=20,
                                    n_fft=int(self.window_length_stft_mfcc * self.sample_rate),
                                    hop_length=int(self.step_size_stft * self.sample_rate), htk=True)
        esp = librosa.feature.melspectrogram(y=audio_n, sr=self.sample_rate,
                                             n_fft=int(self.window_length_stft_esp * self.sample_rate),
                                             hop_length=int(self.step_size_stft * self.sample_rate))
        alto, ancho = MFCC.shape
        self.mfcc = np.reshape(MFCC, (-1, alto, ancho, 1), 'F')
        alto, ancho = esp.shape
        self.esp = np.reshape(esp, (-1, alto, ancho, 1), 'F')

    def predict(self):
        """
        given the mfcc and mel spectrogram the model gives a prediction
        """
        np.argmax(self.model.predict([self.esp, self.mfcc]), axis=-1)

    def listen(self):
        """
        record an audio
        """
        try:
            raw_block = self.stream.read(self.input_frames_per_block, exception_on_overflow=False)
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
