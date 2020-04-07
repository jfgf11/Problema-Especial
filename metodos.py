from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import psutil

import tensorflow.compat.v1 as tf

import matplotlib.pyplot as plt

# Tensor Flow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG

# Recopilacion de datos
import xml.dom.minidom
from scipy.io import wavfile
import numpy as np

# Para el preprocesamiento
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from collections import Counter

import librosa
import progressbar

# Import libraries
import librosa.display
from matplotlib.pyplot import specgram
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os
import sys
import warnings
from scipy import signal
from scipy.fft import fftshift

#Redes neuronales sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.utils.class_weight import compute_class_weight

tf.disable_v2_behavior()

''''
Metodos Google drive 
'''

global d

def autorizacionGoogle():
    gauth = GoogleAuth()

    gauth.LoadCredentialsFile("./credentials/mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("./credentials/mycreds.txt")

    drive = GoogleDrive(gauth)
    return drive

def agregar_hash(root):
    global d
    drive=autorizacionGoogle()
    file_list = drive.ListFile({'q': "'"+str(d[root])+"' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        name=file1['title']
        d[name]=file1['id']
        if '.' not in file1['title']:
            agregar_hash(name)

def hashGoogle():
    drive=autorizacionGoogle()
    global d
    d={}

    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file1 in file_list:

        name = file1['title']
        if name == 'Proyecto Especial Italiano':
            print(name)
            d[name] = file1['id']
            agregar_hash(name)

    print(len(d))
    print('Quedaron guardados')
    return d

def guardarLocalmente(nombre,ruta='./drive/'):
    global d
    drive = autorizacionGoogle()
    id=d[nombre]
    file = drive.CreateFile({'id': id})
    file.GetContentFile(ruta+nombre)

'''''
Recopilación de Datos
Monitoreo del estado de la RAM
Espectograma y recoleccion de variables x, x2 y y
'''

def monitoreo_ram():
    vmem = psutil.virtual_memory()
    if vmem.available < 4000000000:
        return False
    else:
        return True

def dar_ruta(carpeta, features, str_variable, Inicial_pNXML, Final_pNXML, Inicial_nAudios, Final_nAudios,
             ventana_Tiempo,
             sample_rate, Nombre, Sin_Background, Solo_Background, Espectogram_, MFCC_):
    if features:
        Features_or_raw = "_features/"
    else:
        Features_or_raw = "_raw_conv/"
    ventana_Tiempo_String_ = str(ventana_Tiempo).split('.')[1]
    sample_rate_String = str(sample_rate)
    if Sin_Background:
        Back = "_SIN_BACK"
    elif Solo_Background:
        Back = "_SOLO_BACK"
    else:
        Back = ""
    if Espectogram_:
        Esp_o_Mfcc = "Spectopgram"
    elif MFCC_:
        Esp_o_Mfcc = "MFCC"
    else:
        Esp_o_Mfcc = ""

    ruta = carpeta + Features_or_raw + str_variable \
           + str(Inicial_pNXML) + "-" + str(Final_pNXML) + "_XML_" + str(Inicial_nAudios) + "-" + str(Final_nAudios) \
           + "_Audios_" + ventana_Tiempo_String_ + "s_" + sample_rate_String + "_" + Nombre + Back + "_" + Esp_o_Mfcc

    return ruta
    # np.save(ruta, variable)


def Crear_Datos_MFCC_o_Espectogram(pX, sr_, window_length_stft, Step_size_stft, MFCC, Espectogram, ventana_Tiempo):
    contador = 0
    with progressbar.ProgressBar(max_value=(len(pX))) as bar:
        if Espectogram:
            if ventana_Tiempo >= 0.05 and window_length_stft >= 0.025:
                ps = librosa.feature.melspectrogram(y=pX[0], sr=sr_, n_fft=int(window_length_stft * sr_),
                                                    hop_length=int(Step_size_stft * sr_))
            else:
                ps = librosa.feature.melspectrogram(y=pX[0], sr=sr_)
        elif MFCC:
            if ventana_Tiempo >= 0.05 and window_length_stft >= 0.03125:
                ps = librosa.feature.mfcc(y=pX[0], sr=sr_, n_mfcc=13, n_fft=int(window_length_stft * sr_),
                                          hop_length=int(Step_size_stft * sr_), htk=True)
            else:
                ps = librosa.feature.mfcc(y=pX[0], sr=sr_, n_mfcc=13)

        x_2 = np.zeros((len(pX) + 1, len(ps), len(ps[0])))
        for i in range(0, len(pX)):
            if Espectogram:
                if ventana_Tiempo >= 0.05 and window_length_stft >= 0.025:
                    ps = librosa.feature.melspectrogram(y=pX[i], sr=sr_, n_fft=int(window_length_stft * sr_),
                                                        hop_length=int(Step_size_stft * sr_))
                else:
                    ps = librosa.feature.melspectrogram(y=pX[i], sr=sr_)
                ps = librosa.power_to_db(ps, ref=np.max)
            elif MFCC:
                if ventana_Tiempo >= 0.05 and window_length_stft >= 0.03125:
                    ps = librosa.feature.mfcc(y=pX[i], sr=sr_, n_mfcc=13, n_fft=int(window_length_stft * sr_),
                                              hop_length=int(Step_size_stft * sr_), htk=True)
                else:
                    ps = librosa.feature.mfcc(y=pX[i], sr=sr_, n_mfcc=13)
            x_2[i] = ps
            contador += 1
            bar.update(contador)  # Se actualiza la barra de progreso
    x_2 = x_2[0:-1, :]
    return x_2

def ObtenerSonidos(nombre, Inicial_pNXML, Final_pNXML, Inicial_nAudios, Final_nAudios,
                   ventana_Tiempo=0.1, salto_de_ventana=4, Calcular_Features=False, Sin_Background=False,
                   rutaDatosXML="./drive/xml/", rutaDatosParciales="./drive/parciales/",
                   rutaDatosSounds="./drive/sounds/", Solo_Background=False,
                   sample_rate=22050, window_length_stft=0.032, Step_size_stft=0.01,
                   guardarLocal=False, MFCC=False, Espectogram=False):
    fs=0
    NMV = round(ventana_Tiempo * sample_rate)  # Numero de muestras por ventana
    window = np.hamming(NMV)  # se crea la ventana de hamming
    numero_magico = 1350000000  # Depende de la ram del computador 1.35G es maso unas 64GB de ram en el peor caso :v
    z = 0  # Indica si los datos se guardaron en uno o varios archivos. si z se queda en 0 solo se guardo un archivo.
    if guardarLocal:
        datos_x_totales, dato_x2_totales, datos_y_totales = [0], [0], [0]
    else:
        NMV = round(ventana_Tiempo * sample_rate)  # Numero de muestras por ventana
        NMV_advance = round(NMV / salto_de_ventana)  # Numero de muestras por las cuales se avanza
        if Calcular_Features:
            datos_x_totales = np.zeros((1, 312))
        else:
            datos_x_totales = np.zeros((int(numero_magico / NMV), NMV))
        datos_y_totales = np.zeros(int(numero_magico / NMV))
    contador = 0

    with progressbar.ProgressBar(
            max_value=(Final_pNXML - Inicial_pNXML + 1) * (Final_nAudios - Inicial_nAudios + 1)) as bar:
        longitud_actual = 0
        for i in range(Inicial_pNXML, (Final_pNXML + 1)):

            if i < 10:
                h = "0"  # Se hace esto debido que en los audios hay elementos 00001_01 y 00010_1
            else:
                h = ""

            archivo_xml = "000" + h + str(i) + ".xml"
            if guardarLocal:
                guardarLocalmente(archivo_xml, rutaDatosXML)
            else:
                doc = xml.dom.minidom.parse(rutaDatosXML + archivo_xml)
                start = doc.getElementsByTagName(
                    "STARTSECOND")  # Vector que contiene el tiempo en segundos de inicio de todos los eventos
                finish = doc.getElementsByTagName(
                    "ENDSECOND")  # Vector que contiene el tiempo en segundos de finalizacion de todos los eventos
                ID = doc.getElementsByTagName("CLASS_ID")  # Vector que contiene la etiqueta de cada uno de los eventos
                events = doc.getElementsByTagName(
                    "events")  # Indica informacion de todos los eventos en un archivo xml (tamaño)
                a, b, c, d = (events[0].attributes["size"].value)  # Se obtiene el numero de eventos en un audio
                nEventos = int(c + d)  # numero de eventos en un audio

            for a in range(Inicial_nAudios, (Final_nAudios) + 1):  # Numero de audios por xml
                if a < 2:  # Se hace esto debido que en los audios hay elementos 00001_01 y 00010_1
                    r = "0"
                    t = str(a)
                else:
                    t = str(a - 1)
                    r = ""
                archivo_audio = '000' + h + str(i) + '_' + r + t + '.wav'
                if guardarLocal:
                    guardarLocalmente(archivo_audio, rutaDatosSounds)
                else:
                    frameData, fs = librosa.load(rutaDatosSounds + archivo_audio, sr=sample_rate,
                                                 res_type='kaiser_fast')  # Audio seleccionado
                    datos_x = (librosa.util.frame(frameData, frame_length=NMV,
                                                  hop_length=NMV_advance)).T  # Reorganiza los datos dándole saltos de tiempo de NMV_advance y el número de muestras por ventana NMV
                    datos_y = np.zeros(len(
                        frameData))  # Etiquetas de cada uno de los datos, los datos no asignados serán 0 y corresponderan a sonido ambiente

                    for j in range(0, nEventos):  # Se recorre el numero de eventos para cada xml
                        startFrame = float(
                            str(start[j].firstChild.data)) * fs  # Posicion inicial de evento con respecto a frameData
                        endFrame = float(
                            str(finish[j].firstChild.data)) * fs  # Posicion final de evento con respecto a frameData
                        label = ID[j].firstChild.data  # etiqueta del evento
                        datos_y[round(startFrame):round(endFrame)] = int(
                            label)  # Se asigna la etiqueta a cada uno de los datos recopilados

                    datos_y = (((stats.mode(librosa.util.frame(datos_y, frame_length=NMV, hop_length=NMV_advance)))[
                        0]).T)  # Con esto se asigna la etiqueta a datos desplazados en el tiempo
                    datos_y = np.reshape(datos_y, (-1), 'F')

                    if Sin_Background:  # Si se quieren datos sin background
                        datos_x = datos_x[datos_y != 0, :]
                        datos_y = datos_y[datos_y != 0]

                    if Solo_Background:
                        datos_x = (datos_x[datos_y == 0, :])[0:round(len(datos_y[datos_y == 0])), :]
                        datos_y = (datos_y[datos_y == 0])[0:round(len(datos_y[datos_y == 0]))]

                    longitud_siguiente = longitud_actual + len(datos_y)
                    datos_x_totales[longitud_actual:longitud_siguiente, :] = datos_x
                    datos_y_totales[longitud_actual:longitud_siguiente] = datos_y
                    if monitoreo_ram():  # continua ejecucion normal si hay ram suficiente
                        longitud_actual = longitud_siguiente
                    else:  # Se va a acabar la memoria guarda los resultados parciales
                        datos_x_totales = datos_x_totales[0:longitud_siguiente, :]
                        datos_x_totales = datos_x_totales * window  # Se pasa cada uno de los datos por una ventana de hamming
                        datos_y_totales = datos_y_totales[0:longitud_siguiente]
                        datos_y_totales[datos_y_totales == 4] = 1
                        ruta = dar_ruta(rutaDatosParciales, Calcular_Features, 'x_' + str(z) + '_', Inicial_pNXML,
                                        Final_pNXML, Inicial_nAudios,
                                        Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background,
                                        Solo_Background,
                                        False, False)
                        np.save(ruta, datos_x_totales)
                        ruta = dar_ruta(rutaDatosParciales, Calcular_Features, 'y_' + str(z) + '_', Inicial_pNXML,
                                        Final_pNXML, Inicial_nAudios,
                                        Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background,
                                        Solo_Background,
                                        False, False)
                        np.save(ruta, datos_y_totales)

                        if MFCC or Espectogram:  # Se crea el Spectogram para cada dato
                            x2 = Crear_Datos_MFCC_o_Espectogram(datos_x_totales, fs, window_length_stft,
                                                                Step_size_stft, MFCC, Espectogram, ventana_Tiempo)
                            ruta = dar_ruta(rutaDatosParciales, Calcular_Features, 'x2_'+ str(z) + '_', Inicial_pNXML, Final_pNXML,
                                            Inicial_nAudios,
                                            Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background,
                                            Solo_Background, Espectogram, MFCC)
                            np.save(ruta, x2)
                        x2 = None

                        z += 1
                        longitud_actual = 0
                        if Calcular_Features:
                            datos_x_totales = np.zeros((1, 312))
                        else:
                            datos_x_totales = np.zeros((int(numero_magico / NMV), NMV))
                        datos_y_totales = np.zeros(int(numero_magico / NMV))

                contador += 1
                bar.update(contador)  # Se actualiza la barra de progreso

    if not guardarLocal:
        datos_x_totales = datos_x_totales[0:longitud_siguiente, :]
        datos_x_totales = datos_x_totales * window  # Se pasa cada uno de los datos por una ventana de hamming
        datos_y_totales = datos_y_totales[0:longitud_siguiente]
        datos_y_totales[datos_y_totales == 4] = 1

        ruta = dar_ruta(rutaDatosParciales, Calcular_Features, 'x_' + str(z) + '_', Inicial_pNXML, Final_pNXML,
                        Inicial_nAudios,
                        Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
                        False, False)
        np.save(ruta, datos_x_totales)
        ruta = dar_ruta(rutaDatosParciales, Calcular_Features, 'y_' + str(z) + '_', Inicial_pNXML, Final_pNXML,
                        Inicial_nAudios,
                        Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
                        False, False)
        np.save(ruta, datos_y_totales)

        if MFCC or Espectogram:  # Se crea el Spectogram para cada dato
            x2 = Crear_Datos_MFCC_o_Espectogram(datos_x_totales, fs, window_length_stft,
                                                Step_size_stft, MFCC, Espectogram, ventana_Tiempo)
            ruta = dar_ruta(rutaDatosParciales, Calcular_Features, 'x2_'+ str(z) + '_', Inicial_pNXML, Final_pNXML,
                            Inicial_nAudios,Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background,
                            Solo_Background, Espectogram, MFCC)
            np.save(ruta, x2)

        z += 1
        longitud_actual = 0
        datos_x_totales = None
        datos_y_totales = None
        x2 = None

    return z

'''''
Guardar y cargar modelo.
'''

def guardarModelo(pModelo, pRutaModelo, pRutaPesos, pRutaDiagrama):
  modelo_json = pModelo.to_json()

  with open(pRutaModelo, "w") as archivo_json:
      archivo_json.write(modelo_json)

  pModelo.save_weights(pRutaPesos)

  plot_model(pModelo, to_file = pRutaDiagrama, show_shapes = True)

def cargarModelo(pRutaModelo, pRutaPesos):
  archivo_json = open(pRutaModelo, 'r')
  modelo_json = archivo_json.read()
  archivo_json.close()
  modelo = model_from_json(modelo_json)

  modelo.load_weights(pRutaPesos)

  return modelo

'''''
Extraccion de datos juardados y union de ellos.
Preprocesamiento de los datos 
'''

def join(variable,z,rutaDatosParciales,ruta_resultados,Features,Inicial_pNXML,Final_pNXML,Inicial_nAudios,Final_nAudios,
         ventana_Tiempo,sample_rate,nombre,Sin_Background,Solo_Background,MFCC,Espectogram):

    numero_magico = 1350000000  # Depende de la ram del computador 1.35G es maso unas 64GB de ram en el peor caso :v
    NMV = round(ventana_Tiempo * sample_rate)  # Numero de muestras por ventana

    if variable=='x':
        print('mk no, el pc no da, no lo intentes we')

    elif variable=='y':

        print('guardando y')
        y_def = np.zeros(int(numero_magico / NMV))
        anterior=0
        for i in range(z):
            ruta = dar_ruta(rutaDatosParciales, Features, 'y_' + str(i) + '_', Inicial_pNXML, Final_pNXML,
                Inicial_nAudios, Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
                False, False)
            y = np.load(ruta + '.npy')
            print('cargo archivo ',i)

            y_def[anterior:anterior+y.size] = y
            anterior+=y.size
        ruta = dar_ruta(ruta_resultados, Features, 'y_', Inicial_pNXML, Final_pNXML, Inicial_nAudios,
            Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
            False, False)
        y_def=y_def[0:anterior]
        np.save(ruta, y_def)
        y = None
        y_def=None

    elif variable=='x2':

        print('guardando x2')

        x2_def=None
        anterior = 0
        for i in range(z):
            ruta = dar_ruta(rutaDatosParciales, Features, 'x2_' + str(i) + '_', Inicial_pNXML, Final_pNXML,
                            Inicial_nAudios, Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background,
                            Solo_Background,
                            Espectogram, MFCC)
            x2 = np.load(ruta + '.npy')
            print('cargo archivo ', i)
            if i==0:
                x2_def = np.zeros(x2.size*z)
            x2_def[anterior:anterior + x2.size]
            anterior += x2.size
        ruta = dar_ruta(ruta_resultados, Features, 'x2_', Inicial_pNXML, Final_pNXML, Inicial_nAudios,
                        Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
                        Espectogram, MFCC)
        x2_def = x2_def[0:anterior]
        np.save(ruta, x2_def)
        x2 = None
        x2_def = None

    print('----- Guardado -----')

def datosRed2D():
    ruta = dar_ruta()
    Nombre = ''
    y = np.load(ruta + Nombre + ".npy")
    ruta = dar_ruta()
    x2 = np.load(ruta + Nombre + ".npy")

    x_train_2, x_test_2, y_train, y_test = train_test_split(x2, y, random_state=0, test_size=0.20)
    x2 = []
    y = []

    Numero_Datos, alto_2, ancho_2 = x_train_2.shape
    x_train_2 = np.reshape(x_train_2, (-1, 1, alto_2, ancho_2), 'F')
    x_test_2 = np.reshape(x_test_2, (-1, 1, alto_2, ancho_2), 'F')

    pesosClases = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=y_train)
    PesosClases = {0: pesosClases[0] * 0.0001,
                   1: pesosClases[1] + 10,
                   2: pesosClases[2] + 10,
                   3: pesosClases[3] + 30}
    return x_train_2,y_train, x_test_2, y_test, PesosClases

'''''
Creacion y entrenamiento modelo red 2D 
'''

def crearModelo2D(pTasa, pAlpha, pNumFiltros, pTamFiltros, pTamPooling, pNumNeuronas, pOptimizer, T_entrada_1,
                  T_entrada_2):
    modelo = Sequential()

    modelo.add(Input(shape=(1, T_entrada_1, T_entrada_2)))

    modelo.add(Conv2D(pNumFiltros[0], (int(pTamFiltros[0]), int(pTamFiltros[0])), padding='same', activation='relu'))
    modelo.add(Conv2D(pNumFiltros[1], (int(pTamFiltros[1]), int(pTamFiltros[1])), padding='same', activation='relu'))
    modelo.add(Conv2D(pNumFiltros[2], (int(pTamFiltros[2]), int(pTamFiltros[2])), padding='same', activation='relu'))
    modelo.add(Conv2D(pNumFiltros[3], (int(pTamFiltros[3]), int(pTamFiltros[3])), padding='same', activation='relu'))
    # modelo.add(Conv2D(pNumFiltros[4], (int(pTamFiltros[4]),int(pTamFiltros[4])), padding='same', activation = 'relu'))
    # modelo.add(Conv2D(pNumFiltros[5], (int(pTamFiltros[5]),int(pTamFiltros[5])), padding='same', activation = 'relu'))

    modelo.add(MaxPooling2D((int(pTamFiltros[1]), int(pTamFiltros[1])), padding='same'))

    modelo.add(Dropout(0.5))
    modelo.add(Flatten())

    modelo.add(Dense(pNumNeuronas[0], activation='relu'))
    modelo.add(Dense(pNumNeuronas[1], activation='relu'))
    modelo.add(Dense(pNumNeuronas[2], activation='relu'))
    # modelo.add(Dense(pNumNeuronas[3], activation='relu'))

    modelo.add(Dense(4, activation='softmax'))

    sgd = optimizers.SGD(lr=pTasa)  # , momentum=0.9)
    adam = optimizers.Adam(learning_rate=pTasa)
    if pOptimizer == "adam":
        opt = adam
    elif pOptimizer == "sgd":
        opt = sgd
    elif pOptimizer == "rmsprop":
        opt = "rmsprop"

    modelo.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])
    modelo.summary()

    return modelo

def crear_red_2d():
    # Esta celda construye los modelos, a partir de los parametros especificados por cada una de las siguientes variables.
    # Es el numero de filtros que cada capa convolucional utiliza.
    numFiltros = np.array([12, 20, 20, 12, 10, 512])

    # Es el tamaño de los filtros utilizados en cada capa convolucional.
    tamFiltros = np.array([3, 3, 3, 3, 3, 5])

    # Es el tamaño de cada capa de Pooling.
    tamPooling = np.array([4, 2, 3, 3, 3, 3])

    # Es el numero de neuronas en cada capa de la red neuronal que sigue despues de la parte convolucional.
    numNeuronas = np.array([10, 10, 10, 16])

    # Es el tipo de optimizador a utilizar.
    # Se pueden especificar: "sgd", "adam" o "rmsprop"
    optimizer = "rmsprop"

    # Es la tasa de aprendizaje del optimizador.
    tasa = 0.1

    # Es el parametro de regularizacion a utilizar.
    alpha = 0.01

    x_train_2,y_train,x_test_2,y_test,pesos=datosRed2D()

    y_train=None
    x_test_2=None
    y_test=None
    pesos = None

    Numero_Datos, uno, alto_2, ancho_2 = x_train_2.shape

    modelo2 = crearModelo2D(tasa, alpha, numFiltros, tamFiltros, tamPooling, numNeuronas, optimizer, T_entrada_1=alto_2,
                            T_entrada_2=ancho_2)

    # Esta linea muestra un diagrama de la red neuronal.
    SVG(model_to_dot(modelo2, show_shapes=True, expand_nested=True, dpi=50).create(prog='dot', format='svg'))


def entrenarRed(modelo):
    epocas = 100
    batchSize = 5000
    x_train_2,y_train,x_test_2,y_test,pesos=datosRed2D()
    # modelo1.compile(loss='sparse_categorical_crossentropy', optimizer = "rmsprop", metrics = ['sparse_categorical_accuracy'])

    for i in range(0, 1):
        # hist = modelo1.fit(x, y, verbose = 1, validation_data=(x, y), epochs = epocas, batch_size = batchSize)#, class_weight = pesosClases)
        hist = modelo.fit(x_train_2, y_train, validation_data=(x_test_2, y_test), epochs=epocas, batch_size=batchSize,
                           class_weight=pesos)


''''
Visualización de resultados
'''

def graficarMatrizConfusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))

    ax = sns.heatmap(cm, annot=True, cbar=False);

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.xlabel("Clase Prediccion")
    plt.ylabel("Clase Verdadera")
    plt.title("Matriz de Confusion")

    plt.show()