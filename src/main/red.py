#!/usr/bin/env python3
from tensorflow.keras.models import model_from_json
import numpy as np

def cargarModelo(pRutaModelo, pRutaPesos):
  archivo_json = open(pRutaModelo, 'r')
  modelo_json = archivo_json.read()
  archivo_json.close()
  modelo = model_from_json(modelo_json)
  modelo.load_weights(pRutaPesos)
  return modelo

rutaModelo=".Modelo_Male_200 (1).json"
rutaPesos=".Pesos_Modelo_Male_200 (1).h5"
SAMPLE_RATE=22050
window_length_stft = 0.025
Step_size_stft = 0.010
ventana_Tiempo_ = 0.050
#pX[i] DATOS DE AUDIO

modelo1=cargarModelo(rutaModelo,rutaPesos)

modelo1.compile(loss='sparse_categorical_crossentropy', optimizer = "rmsprop", metrics = ['sparse_categorical_accuracy'])
modelo1.predict_classes(x_test_2)

mfcc=librosa.feature.mfcc(y=pX[i], sr=SAMPLE_RATE, n_mfcc=13)#,n_fft = int(window_length_stft*SAMPLE_RATE), hop_length = int(Step_size_stft*SAMPLE_RATE), htk=True )
modelo1.predict_classes(mfcc)
np.argmax(modelo1.predict(mfcc), axis=-1)
