#!/usr/bin/env python3
import pyaudio
import struct
import numpy as np
import threading
import time
import librosa
from tensorflow.keras.models import model_from_json
from termcolor import colored


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

    def processBlock(self, p_audio):
        """
        process an audio signal and gives a prediction
        """
        self.preprocesing(p_audio)
        self.predict()

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
        clase = np.argmax(self.model.predict([self.esp, self.mfcc]), axis=-1)
        if clase[0] == 0:
            print(colored('0', 'magenta'))
        elif clase[0] == 1:
            print(colored('1', 'green'))
        elif clase[0] == 2:
            print(colored('2', 'blue'))
        elif clase[0] == 3:
            print(colored('3', 'red'))

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
    audio = AudioHandler()
    run = True
    threads = False
    while run:
        toProcess = audio.listen()
        if threads:
            # print(threading.active_count())
            x = threading.Thread(target=audio.processBlock, args=(toProcess,))
            x.start()
            if threading.active_count() > 10:
                run = False
        else:
            audio.preprocesing(toProcess)
            audio.predict()

    while threading.active_count() > 0:
        print('Cerrando Threads')
        time.sleep(0.5)

    print('Se han cerrado todos los threads')
