#!/usr/bin/env python3
from tensorflow.keras.models import model_from_json

def cargarModelo(pRutaModelo, pRutaPesos):
  archivo_json = open(pRutaModelo, 'r')
  modelo_json = archivo_json.read()
  archivo_json.close()
  modelo = model_from_json(modelo_json)

  modelo.load_weights(pRutaPesos)

  return modelo

modelo1.compile(loss='sparse_categorical_crossentropy', optimizer = "rmsprop", metrics = ['sparse_categorical_accuracy'])
rutaModelo=".Modelo_Jorge_1.json"
rutaPesos=".Pesos_Modelo_Jorge_1.h5"

modelo=cargarModelo(rutaModelo,rutaPesos)

