import statistics

from IPython.display import SVG
from keras.backend import clear_session
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import psutil

# Tensor Flow
import tensorflow.compat.v1  as tf
tf.disable_v2_behavior()
import logging
logging.getLogger('tensorflow').disabled = True
from tensorflow_core import metrics

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D, BatchNormalization, \
    Activation, concatenate, AveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.initializers import glorot_uniform
# Recopilacion de datos
import xml.dom.minidom
import numpy as np
# Para el preprocesamiento
import librosa
import progressbar
# Import libraries
import librosa.display
# Redes neuronales sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import collections
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import csv
from notify_run import Notify
import time

from bayes_opt import BayesianOptimization

import os

os.environ["PATH"] += os.pathsep + 'C:/Users/TAURET/Documents/graphviz/release/bin'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
-------------------------------------------------------------------------------------------------------------------
--------------------------------------------------- Metodos Google drive  -----------------------------------------
-------------------------------------------------------------------------------------------------------------------
"""

global d
global drive


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
    global drive
    file_list = drive.ListFile({'q': "'" + str(d[root]) + "' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        name = file1['title']
        d[name] = file1['id']
        if '.' not in file1['title']:
            agregar_hash(name)


def hashGoogle():
    global drive
    global d

    drive = autorizacionGoogle()
    d = {}

    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file1 in file_list:

        name = file1['title']
        if name == 'Proyecto Especial':
            print(name)
            d[name] = file1['id']
            agregar_hash(name)

    print(len(d))
    print('Quedaron guardados')
    return d


def guardarLocalmente(nombre, ruta='./drive/'):
    global d
    drive = autorizacionGoogle()
    id = d[nombre]
    file = drive.CreateFile({'id': id})
    file.GetContentFile(ruta + nombre)


"""
-------------------------------------------------------------------------------------------------------------------
--------------------------------------------- Recopilación de Datos  ----------------------------------------------
--------------------------------------- Monitoreo del estado de la RAM  -------------------------------------------
------------------------------ Espectograma y recoleccion de variables x, x2 y y  ---------------------------------
-------------------------------------------------------------------------------------------------------------------
"""


def monitoreo_ram():
    vmem = psutil.virtual_memory()
    if vmem.available < 4000000000:
        return False
    else:
        return True


def dar_ruta(carpeta, str_variable, salto, Inicial_pNXML, Final_pNXML, ventana_Tiempo, sample_rate, Sin_Background,
             Solo_Background, Espectogram_, MFCC_):
    sample_rate_String = str(sample_rate)
    if str_variable == 'y_':
        MFCC_, Espectogram_ = False, False
    if Sin_Background:
        Back = "_SIN_BACK"
    elif Solo_Background:
        Back = "_SOLO_BACK"
    else:
        Back = ""
    if Espectogram_:
        Esp_o_Mfcc = "_spectrogram"
    elif MFCC_:
        Esp_o_Mfcc = "_MFCC"
    else:
        Esp_o_Mfcc = ""

    if str_variable == 'y_':
        ruta = carpeta + str_variable + 'salto_' + str(salto) + '_' + str(Inicial_pNXML) + "-" + str(Final_pNXML) + \
               "_Audios_" + ventana_Tiempo + "ms_" + sample_rate_String + Back + Esp_o_Mfcc
    else:
        ruta = carpeta + str_variable + 'salto_' + str(salto) + '_' + str(Inicial_pNXML) + "-" + str(
            Final_pNXML) + "_Audios_" + ventana_Tiempo \
               + "ms_" + sample_rate_String + Back + Esp_o_Mfcc
    return ruta


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
            if ventana_Tiempo >= 0.05 and window_length_stft >= 0.100:
                ps = librosa.feature.mfcc(y=pX[0], sr=sr_, n_mfcc=20, n_fft=int(window_length_stft * sr_),
                                          hop_length=int(Step_size_stft * sr_), htk=True)
            else:
                ps = librosa.feature.mfcc(y=pX[0], sr=sr_, n_mfcc=20)

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
                if ventana_Tiempo >= 0.05 and window_length_stft >= 0.100:
                    ps = librosa.feature.mfcc(y=pX[i], sr=sr_, n_mfcc=20, n_fft=int(window_length_stft * sr_),
                                              hop_length=int(Step_size_stft * sr_), htk=True)
                else:
                    ps = librosa.feature.mfcc(y=pX[i], sr=sr_, n_mfcc=20)
            x_2[i] = ps
            contador += 1
            bar.update(contador)  # Se actualiza la barra de progreso
    x_2 = x_2[0:-1, :]
    return x_2


def ObtenerSonidos(Inicial_pNXML, Final_pNXML, ventana_Tiempo=0.1, salto_de_ventana=4, Sin_Background=False,
                   rutaDatosXML="./drive/xml/", ruta_datos="./drive/parciales/",
                   rutaDatosSounds="./drive/sounds/", Solo_Background=False,
                   sample_rate=22050, window_length_stft=0.032, Step_size_stft=0.01,
                   guardarLocal=False):
    ventana_Tiempo_str = '450'
    if not guardarLocal:
        NMV = round(ventana_Tiempo * sample_rate)  # Numero de muestras por ventana
        NMV_advance = round(NMV / salto_de_ventana)  # Numero de muestras por las cuales se avanza
        # 190 representa que cada audio dura aproximadamente 3 minutos y 10 segundos
        numero_datos = int(((190 / ventana_Tiempo) * salto_de_ventana) * (Final_pNXML - Inicial_pNXML + 1))
        datos_x_totales = np.zeros((numero_datos, NMV))
        datos_y_totales = np.zeros(numero_datos)
    contador = 0

    with progressbar.ProgressBar(max_value=(Final_pNXML - Inicial_pNXML + 1)) as bar:
        longitud_actual = 0
        for i in range(Inicial_pNXML, (Final_pNXML + 1)):
            if i < 10:
                h = "0"  # Se hace esto debido que en los audios hay elementos 00001_01 y 00010_1
            else:
                h = ""
            archivo_xml = "00" + h + str(i) + ".xml"
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

            archivo_audio = '00' + h + str(i) + '.wav'  # cada xml solo tiene un audio asociado.
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
                longitud_actual = longitud_siguiente

                contador += 1
                bar.update(contador)  # Se actualiza la barra de progreso

    if not guardarLocal:
        datos_x_totales = datos_x_totales[0:longitud_siguiente, :]
        datos_y_totales = datos_y_totales[0:longitud_siguiente]

        ruta = dar_ruta(ruta_datos, 'y_', salto_de_ventana, Inicial_pNXML, Final_pNXML, ventana_Tiempo_str, sample_rate,
                        Sin_Background, Solo_Background, False, False)
        np.save(ruta, datos_y_totales)

        datos_y_totales = None
        x2 = None

        x2 = Crear_Datos_MFCC_o_Espectogram(datos_x_totales, fs, window_length_stft,
                                            Step_size_stft, False, True, ventana_Tiempo)
        ruta = dar_ruta(ruta_datos, 'x2_', salto_de_ventana, Inicial_pNXML, Final_pNXML, ventana_Tiempo_str,
                        sample_rate,
                        Sin_Background, Solo_Background, True, False)
        np.save(ruta, x2)

        x2 = None

        x2 = Crear_Datos_MFCC_o_Espectogram(datos_x_totales, fs, window_length_stft,
                                            Step_size_stft, True, False, ventana_Tiempo)
        ruta = dar_ruta(ruta_datos, 'x2_', salto_de_ventana, Inicial_pNXML, Final_pNXML, ventana_Tiempo_str,
                        sample_rate,
                        Sin_Background, Solo_Background, False, True)
        np.save(ruta, x2)

        longitud_actual = 0
        datos_x_totales = None
        x2 = None


"""
-------------------------------------------------------------------------------------------------------------------
------------------------------------------------- Guardar y cargar modelo -----------------------------------------
-------------------------------------------------------------------------------------------------------------------
"""


def guardarModelo(pModelo, pRutaModelo, pRutaPesos):
    modelo_json = pModelo.to_json()

    with open(pRutaModelo, "w") as archivo_json:
        archivo_json.write(modelo_json)

    pModelo.save_weights(pRutaPesos)


def cargarModelo(pRutaModelo, pRutaPesos):
    archivo_json = open(pRutaModelo, 'r')
    modelo_json = archivo_json.read()
    archivo_json.close()
    modelo = model_from_json(modelo_json)
    modelo.load_weights(pRutaPesos)
    return modelo


"""
-------------------------------------------------------------------------------------------------------------------
---------------------------------------------- Preprocesamiento de los datos---------------------------------------
-------------------------------------------------------------------------------------------------------------------
"""


def extrar_datos(Inicial_pNXML, Final_pNXML, Espectrogram, MFCC, carpeta,
                 ventana_Tiempo_string="690", sample_rate=22050, Sin_Background=False, Solo_Background=False,
                 salto=4):
    x_spect = None
    x_MFCC = None
    y = None
    if Espectrogram:
        ruta = dar_ruta(carpeta, 'x2_', salto, Inicial_pNXML, Final_pNXML,
                        ventana_Tiempo_string, sample_rate, Sin_Background, Solo_Background, True, False)

        x_spect = np.load(ruta + ".npy")
    if MFCC:
        ruta = dar_ruta(carpeta, 'x2_', salto, Inicial_pNXML, Final_pNXML,
                        ventana_Tiempo_string, sample_rate, Sin_Background, Solo_Background, False, True)

        x_MFCC = np.load(ruta + ".npy")
    ruta = dar_ruta(carpeta, 'y_', salto, Inicial_pNXML, Final_pNXML,
                    ventana_Tiempo_string, sample_rate, Sin_Background, Solo_Background, True, False)

    y = np.load(ruta + ".npy")
    return x_spect, x_MFCC, y


def reshape_data(x_train, x_test):
    numero_Datos, alto, ancho = x_train.shape
    x_train = np.reshape(x_train, (-1, alto, ancho, 1), 'F')
    x_test = np.reshape(x_test, (-1, alto, ancho, 1), 'F')
    return x_train, x_test


def train_test(x2, y_train):
    x2, x2_test, y_train, y_test = train_test_split(x2, y_train, random_state=0, test_size=0.001)
    return x2, x2_test, y_train, y_test


"""
-------------------------------------------------------------------------------------------------------------------
------------------------------------------------- Metodos de balanceo ---------------------------------------------
-------------------------------------------------------------------------------------------------------------------
"""


def random_under_sampling(x2, y_train):
    numero_datos, alto, ancho = x2.shape
    x2 = np.reshape(x2, (-1, alto * ancho), 'F')

    rus = RandomUnderSampler(random_state=42)
    x2, y_train = rus.fit_resample(x2, y_train)

    x2 = np.reshape(x2, (-1, alto, ancho), 'F')

    return x2, y_train


def random_over_sampling(x2, y_train):
    numero_datos, alto, ancho = x2.shape
    x2 = np.reshape(x2, (-1, alto * ancho), 'F')

    ros = RandomOverSampler(sampling_strategy='not majority', random_state=0)
    x2, y_train = ros.fit_resample(x2, y_train)

    x2 = np.reshape(x2, (-1, alto, ancho), 'F')

    return x2, y_train


def pesos(y_p):
    return compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=y_p)


def over_under_pesos(pUnder_over_pesos, esp, mfcc, x2_esp, x2_mfcc, y_train):
    if pUnder_over_pesos == 'undersampling':
        if mfcc:
            x2_mfcc, y_train_mfcc = random_under_sampling(x2_mfcc, y_train)
            pesosClases = pesos(y_train_mfcc)
            y_train_gl = y_train_mfcc
        if esp:
            x2_esp, y_train_esp = random_under_sampling(x2_esp, y_train)
            pesosClases = pesos(y_train_esp)
            y_train_gl = y_train_esp
    elif pUnder_over_pesos == 'oversampling':
        if mfcc:
            x2_mfcc, y_train_mfcc = random_over_sampling(x2_mfcc, y_train)
            pesosClases = pesos(y_train_mfcc)
            y_train_gl = y_train_mfcc
        if esp:
            x2_esp, y_train_esp = random_over_sampling(x2_esp, y_train)
            pesosClases = pesos(y_train_esp)
            y_train_gl = y_train_esp
    elif pUnder_over_pesos == 'pesos':
        pesosClases = pesos(y_train)
        y_train_gl = y_train

    return x2_esp, x2_mfcc, y_train_gl, pesosClases


"""
-------------------------------------------------------------------------------------------------------------------
------------------------------------------------- Modelos Convolucionales -----------------------------------------
-------------------------------------------------------------------------------------------------------------------
"""


def crearModelo_2D_Doble(pTasa, pNumNeuronas, T_entrada_1, T_entrada_2, T_entrada_3, T_entrada_4):
    # --------------------------------------------------------------
    # ------------------ CNN MFCC ----------------------------------
    # --------------------------------------------------------------
    capaEntrada_1 = tf.keras.layers.Input(shape=(T_entrada_3, T_entrada_4, 1))

    # Capa 1
    X = tf.keras.layers.ZeroPadding2D(padding=(0, 1))(capaEntrada_1)
    X = tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 2
    X = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 3
    X = tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(X)

    # Capa 4
    # X = tf.keras.layers.Conv2D(pNumFiltros[3], int(pTamFiltros[3]), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = tf.keras.layers.Activation('relu')(X)
    flatten_1 = tf.keras.layers.Flatten()(X)

    # --------------------------------------------------------------
    # ------------------ CNN Spec ---------------------------------
    # --------------------------------------------------------------

    capaEntrada_2 = tf.keras.layers.Input(shape=(T_entrada_1, T_entrada_2, 1))

    # Capa 1
    X = tf.keras.layers.Conv2D(kernel_size=5, filters=6, kernel_initializer=glorot_uniform(seed=0))(capaEntrada_2)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 2
    X = tf.keras.layers.Conv2D(kernel_size=5, filters=16, kernel_initializer=glorot_uniform(seed=0))(capaEntrada_2)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 3
    # X = tf.keras.layers.Conv2D(pNumFiltros[2], int(pTamFiltros[2],), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = tf.keras.layers.Activation('relu')(X)
    # X = tf.keras.layers.MaxPooling2D(int(pTamPooling[2]), padding='same')(X)

    flatten_2 = tf.keras.layers.Flatten()(X)

    capas = tf.keras.layers.concatenate([flatten_1, flatten_2])

    # capas = Dropout(0.5)(capas)
    n1 = int(pNumNeuronas[0])
    n2 = int(pNumNeuronas[1])
    n3 = int(pNumNeuronas[2])
    capas = tf.keras.layers.Dense(n1, activation='relu')(capas)
    capas = tf.keras.layers.Dense(n2, activation='relu')(capas)
    capas = tf.keras.layers.Dense(n3, activation='relu')(capas)

    capaSalida = tf.keras.layers.Dense(4, activation='softmax')(capas)

    modelo = tf.keras.models.Model(inputs=[capaEntrada_1, capaEntrada_2], outputs=capaSalida)

    return modelo


"""
-------------------------------------------------------------------------------------------------------------------
------------------------------------------ Metodos entrenamiento y validacion -------------------------------------
-------------------------------------------------------------------------------------------------------------------
"""


def join(variable, lista):
    ans = None
    if variable == 'y_':
        y_def = None
        anterior = 0
        for i in range(len(lista)):
            y = np.load(lista[i])

            if i == 0:
                y_def = np.zeros(y.size * 2 * (len(lista)))

            y_def[anterior:anterior + y.size] = y
            anterior += y.size
        y_def = y_def[0:anterior]
        ans = y_def
        y = None
        y_def = None

    elif variable == 'x2_':
        x2_def = None
        anterior = 0
        for i in range(len(lista)):
            x2 = np.load(lista[i])
            if i == 0:
                x2_def = np.zeros([x2.shape[0] * 2 * (len(lista)), x2.shape[1], x2.shape[2]])
            x2_def[anterior:anterior + x2.shape[0], :, :] = x2
            anterior += x2.shape[0]
        x2_def = x2_def[0:anterior, :, :]
        ans = x2_def
        x2 = None
        x2_def = None

    return ans


def graficarMatrizConfusion(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))

    ax = sns.heatmap(cm, annot=True, cbar=False);

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.xlabel("Clase Prediccion")
    plt.ylabel("Clase Verdadera")
    plt.title("Matriz de Confusion" + titulo)

    plt.show()


def crear_folds(numero_audios, numero_folds):
    base_audios = int(numero_audios / numero_folds)
    numero_bases_con_un_audio_mas = numero_audios % numero_folds
    audios = [0]
    cuenta = 0
    for i in range(numero_folds):
        if i < numero_bases_con_un_audio_mas:
            cuenta += base_audios + 1
            audios.append(cuenta)
        else:
            cuenta += base_audios
            audios.append(cuenta)
    return audios


def listas(folds, i, ruta_resultados, salto, ventana_tiempo_string,
           sample_rate, Sin_Background_, Solo_Background_,
           esp, mfcc, numero_audios):
    lista_y = []
    lista_x2 = []
    lista_y_validacion = []
    lista_x2_validacion = []
    lista_x2_mfcc = []
    lista_x2_esp = []
    lista_x2_esp_validacion = []
    lista_x2_mfcc_validacion = []

    for j in range(1, folds[i] + 1):

        ruta = dar_ruta(ruta_resultados, 'y_', salto, j, j, ventana_tiempo_string,
                        sample_rate, Sin_Background_, Solo_Background_,
                        esp, mfcc)
        lista_y.append(ruta + '.npy')
        if esp and mfcc:
            ruta_esp = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                                sample_rate, Sin_Background_, Solo_Background_,
                                esp, False)
            lista_x2_esp.append(ruta_esp + '.npy')
            ruta_mfcc = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                                 sample_rate, Sin_Background_, Solo_Background_,
                                 False, mfcc)
            lista_x2_mfcc.append(ruta_mfcc + '.npy')
        else:
            ruta = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                            sample_rate, Sin_Background_, Solo_Background_,
                            esp, mfcc)
            lista_x2.append(ruta + '.npy')

    for j in range(folds[i] + 1, folds[i + 1] + 1):
        ruta = dar_ruta(ruta_resultados, 'y_', salto, j, j, ventana_tiempo_string,
                        sample_rate, Sin_Background_, Solo_Background_,
                        esp, mfcc)
        lista_y_validacion.append(ruta + '.npy')
        if esp and mfcc:
            ruta_esp = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                                sample_rate, Sin_Background_, Solo_Background_,
                                esp, False)
            lista_x2_esp_validacion.append(ruta_esp + '.npy')
            ruta_mfcc = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                                 sample_rate, Sin_Background_, Solo_Background_,
                                 False, mfcc)
            lista_x2_mfcc_validacion.append(ruta_mfcc + '.npy')
        else:
            ruta = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                            sample_rate, Sin_Background_, Solo_Background_,
                            esp, mfcc)
            lista_x2_validacion.append(ruta + '.npy')

    for j in range(folds[i + 1] + 1, numero_audios + 1):
        ruta = dar_ruta(ruta_resultados, 'y_', salto, j, j, ventana_tiempo_string,
                        sample_rate, Sin_Background_, Solo_Background_,
                        esp, mfcc)
        lista_y.append(ruta + '.npy')
        if esp and mfcc:
            ruta_esp = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                                sample_rate, Sin_Background_, Solo_Background_,
                                esp, False)
            lista_x2_esp.append(ruta_esp + '.npy')
            ruta_mfcc = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                                 sample_rate, Sin_Background_, Solo_Background_,
                                 False, mfcc)
            lista_x2_mfcc.append(ruta_mfcc + '.npy')
        else:
            ruta = dar_ruta(ruta_resultados, 'x2_', salto, j, j, ventana_tiempo_string,
                            sample_rate, Sin_Background_, Solo_Background_,
                            esp, mfcc)
            lista_x2.append(ruta + '.npy')

    return lista_y, lista_x2, lista_y_validacion, lista_x2_validacion, lista_x2_mfcc, lista_x2_esp, lista_x2_esp_validacion, lista_x2_mfcc_validacion


def entrenar(epocas, batch_size, modelo, x_train_2, y_train, x_test_2, y_test, pesos, titulo, Espectrogram=True,
             MFCC=False, x_train_1=None, x_test_1=None):
    if Espectrogram and MFCC:
        if batch_size != -1:
            hist = modelo.fit([x_train_1, x_train_2], y_train, validation_data=([x_test_1, x_test_2], y_test),
                              epochs=epocas, batch_size=batch_size, class_weight=pesos, verbose=2)
        else:
            hist = modelo.fit([x_train_1, x_train_2], y_train, validation_data=([x_test_1, x_test_2], y_test),
                              epochs=epocas, class_weight=pesos, verbose=2)

        # evaluate the model
        _, train_acc = modelo.evaluate([x_train_1, x_train_2], y_train, verbose=0)
        _, test_acc = modelo.evaluate([x_test_1, x_test_2], y_test, verbose=0)

        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        # plot loss during training
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='test')
        plt.legend()
        # plot accuracy during training
        plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(hist.history['sparse_categorical_accuracy'], label='train')
        plt.plot(hist.history['val_sparse_categorical_accuracy'], label='test')
        plt.legend()
        plt.show()
        # plot F1 score
        # plt.subplot(313)
        y_entrenamiento = f1_score(y_train, (modelo.predict([x_train_1, x_train_2])).argmax(axis=-1), average='macro')
        y_validacion = f1_score(y_test, (modelo.predict([x_test_1, x_test_2])).argmax(axis=-1), average='macro')
        print('F1 score entrenamiento: ' + str(y_entrenamiento), 'F1 score validación: ' + str(y_validacion))
        # plt.plot(y_entrenamiento, label='F1 Entrenamiento')
        # plt.plot(y_validacion, label='F1 Validacion')
        # plt.xlabel('Epoca')
        # plt.ylabel('F1_Score')
        # plt.title("F1_score vs Epoca")
        # plt.legend()
        # plt.show()
    else:
        if batch_size != -1:
            hist = modelo.fit(x_train_2, y_train, validation_data=(x_test_2, y_test), epochs=epocas,
                              batch_size=batch_size, class_weight=pesos, verbose=2)
        else:
            hist = modelo.fit(x_train_2, y_train, validation_data=(x_test_2, y_test), epochs=epocas, class_weight=pesos,
                              verbose=2)

    # print(x_train_2.shape)
    # predict = modelo.predict(x_test_2)
    # y_pred= np.argmax(predict, axis=-1)
    # y_pred = modelo.predict_classes(x_test_2)
    # graficarMatrizConfusion(y_test, y_pred, titulo)

    return modelo, train_acc, test_acc


def resultados(conf_matrix_array, titulo):
    arr = np.mean(conf_matrix_array, axis=0)
    df_cm = pd.DataFrame(arr[0], index=[i for i in "0123"],
                         columns=[i for i in "0123"])
    plot = plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True)
    plt.xlabel("Clase Prediccion")
    plt.ylabel("Clase Verdadera")
    plt.title("Matriz de Confusion lr: " + titulo)
    plt.show()
    plot.savefig('./drive/modelos/matriz/VoltageConfMatrixR.png')


def evaluate_network(lr, bs, dense1, dense2, dense3):
    # Cross validation
    numero_audios = 24
    numero_folds = 8
    under_over_pesos = 'oversampling'

    ventana_tiempo_string = "450"
    esp = True
    mfcc = True

    datos = [[22050, 4]]
    iteraciones = 1  # Número de iteraciones
    ruta_resultados = './drive/datos_procesados/'

    Sin_Background_ = False  # True: no se obtienen datos de background; False: No se obtienen datos de background  NO MOVER
    Solo_Background_ = False  # Solo obtener datos de background  NO MOVER

    titulo = ''

    mean_benchmark = [0]
    epochs_needed = [0]
    epocas = 1
    batch_size = int(round(bs))
    tasa = lr
    conf_matrix_array = []
    numNeuronas = np.array([dense1, dense2, dense3])

    folds = crear_folds(numero_audios, numero_folds)
    salto = datos[0][1]
    sample_rate = datos[0][0]

    scores = []

    for i in range(numero_folds):
        lista_y, lista_x2, lista_y_validacion, lista_x2_validacion, lista_x2_mfcc, lista_x2_esp, lista_x2_esp_validacion, lista_x2_mfcc_validacion = listas(
            folds, i, ruta_resultados, salto, ventana_tiempo_string, sample_rate, Sin_Background_, Solo_Background_,
            esp, mfcc, numero_audios)

        y_train = join('y_', lista_y)
        y_test = join('y_', lista_y_validacion)

        x2_esp = join('x2_', lista_x2_esp)
        x2_mfcc = join('x2_', lista_x2_mfcc)
        x2_esp_validacion = join('x2_', lista_x2_esp_validacion)
        x2_mfcc_validacion = join('x2_', lista_x2_mfcc_validacion)
        x2_esp, x2_mfcc, y_train, pesosClases = over_under_pesos(under_over_pesos, esp, mfcc, x2_esp, x2_mfcc,
                                                                 y_train)

        x2_esp, x2_esp_validacion = reshape_data(x2_esp, x2_esp_validacion)
        x2_mfcc, x2_mfcc_validacion = reshape_data(x2_mfcc, x2_mfcc_validacion)

        numero_datos, alto_1_1, ancho_1_2, uno = x2_esp.shape
        numero_datos, alto_2_1, ancho_2_2, uno = x2_mfcc.shape
        modelo_doble = crearModelo_2D_Doble(tasa, numNeuronas, T_entrada_1=alto_2_1, T_entrada_2=ancho_2_2,
                                            T_entrada_3=alto_1_1, T_entrada_4=ancho_1_2)

        # opt = tf.keras.optimizers.RMSprop(lr=0.00033)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        modelo_doble.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                             metrics=['sparse_categorical_accuracy'])


        modelo_doble = entrenar(epocas, batch_size, modelo_doble, x2_mfcc,
                                y_train, x2_mfcc_validacion, y_test,
                                pesosClases, titulo + ' entrenamiento',
                                esp, mfcc, x2_esp, x2_esp_validacion)

        x=[x2_esp_validacion, x2_mfcc_validacion]
        y_prob = modelo_doble.predict(x)
        y_pred = y_prob.argmax(axis=-1)
        # confusion_matrix = graficarMatrizConfusion(y_test, y_pred, titulo + j)

        confusion_matrix_ = confusion_matrix(y_test, y_pred)
        confusion_matrix_ = confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]
        conf_matrix_array.append([confusion_matrix_])
        print(conf_matrix_array)

        _, acc = modelo_doble.evaluate([x2_esp_validacion, x2_mfcc_validacion], y_test, verbose=0)
        scores.append(acc)
        # Predict on the out of boot (validation)
        pred = modelo_doble.predict_classes([x2_esp_validacion, x2_mfcc_validacion])
        # Measure this bootstrap's log loss
        y_compare = np.argmax(y_test, axis=1)  # For log loss calculation
        score = metrics.log_loss(y_compare, pred)
        mean_benchmark.append(score)
        m1 = statistics.mean(mean_benchmark)
        m2 = statistics.mean(epochs_needed)
        mdev = statistics.pstdev(mean_benchmark)
        time_took = time.time() - start_time

    clear_session()
    return (-m1)


"""
-------------------------------------------------------------------------------------------------------------------
----------------------------------------------- Ejecucion del programa --------------------------------------------
-------------------------------------------------------------------------------------------------------------------
"""

while True:
    print('Menu para seleccionar la accion a ejecutar')
    print('-------------------------------------------')
    print('1. Descargar Audios y xml de google drive.')
    print('2. Sacar x2 y y')
    print('3. Descargar modelo de Drive')
    print('4. Guardar modelo')
    print('5. Red 2d')
    print('6. Red doble 2D')
    print('7. Bayesian optimization')
    print('8. Salir')
    print('-------------------------------------------')
    print('Porfavor seleccione una opción')
    print('CUIDADO: algunas de estas operaciones consumen bastante '
          'memoria y pueden trabar su computador, antes de correr algo '
          'este seguro de que cuenta con la memoria suficiente !!!')

    opcion = int(input())

    while not (0 < opcion < 10):
        print('Porfavor seleccione una opción entre 1 y 8')
        opcion = int(input())

    print('Usted selecciono la opción: ', opcion)

    """
    -------------------------------------------------------------------------------------------------------------------
    ----------------------------------------------- Ejecucion del programa --------------------------------------------
    -------------------------------------------------------------------------------------------------------------------
    """

    if True:  # parametros
        # Cross validation
        numero_audios = 24
        numero_folds = 8
        under_over_pesos = 'oversampling'

        ventana_Tiempo_ = 0.450  # La ventana de tiempo de cada muestra (XX_s)
        ventana_tiempo_string = "450"
        Inicial_pNXML_ = 1  # Número inicial de archivos XML utilizados
        Final_pNXML_ = 24  # Número final de archivos XML utilizados
        inicial_pNXML_validacion = 1
        Final_pNXML_validacion = 23
        esp = True
        mfcc = True

        # indica los datos con los que se va a entrenar [sample rate, salto de ventana]
        datos = [[22050, 4]]
        # Se ponen los metodos de balanceo a usar. ('oversampling', 'undersampling', 'pesos')
        # under_over_pesos = ['oversampling']
        iteraciones = 1  # Número de iteraciones

        rutaDatosXML_ = "./drive/xml/"  # Ruta para encontrar archivos xml  NO MOVER
        rutaDatosSounds_ = "./drive/sounds/"  # Ruta para encontrar Audios  NO MOVER
        ruta_resultados = './drive/datos_procesados/'

        Frecuencia_Corte = 11000  # Se define la frecuencia de Corte en Hz, la frecuencia minima que tiene sentido es 4000, si se quiere que el filtro no haga nada setear a 11000Hz
        Sin_Background_ = False  # True: no se obtienen datos de background; False: No se obtienen datos de background  NO MOVER
        Solo_Background_ = False  # Solo obtener datos de background  NO MOVER
        window_length_stft_ = 0.032  # Ventana de tiempo para la short-Time Fourier Transform
        Step_size_stft_ = 0.010  # Saltos el en tiempo para la transformada de Fourier, fíjenlo menor a la ventana stft, si quieren pueden aumentar

        titulo = ''

        # --------------------------------------------------------------
        # ---------------------- Parametros Red-------------------------
        # --------------------------------------------------------------

        # Esta celda construye los modelos, a partir de los parametros especificados por cada una de las siguientes variables.
        # Es el numero de filtros que cada capa convolucional utiliza.
        numFiltros = np.array([10, 30, 50, 128, 10, 10])

        # Es el tamaño de los filtros utilizados en cada capa convolucional.
        tamFiltros = np.array([3, 5, 7, 3, 2, 5])

        # Es el tamaño de cada capa de Pooling.
        tamPooling = np.array([2, 2, 2, 2, 2, 2])

        # Es el numero de neuronas en cada capa de la red neuronal que sigue despues de la parte convolucional.
        numNeuronas = np.array([20, 20, 20, 16])

        # Es el tipo de optimizador a utilizar.
        # Se pueden especificar: "sgd", "adam" o "rmsprop"
        optimizer = "adam"

        # Es la tasa de aprendizaje del optimizador.
        lr = 0.2

        # Es el parametro de regularizacion a utilizar.
        alpha = 0.01

        epocas = 15
        batch_size = 500

    if opcion == 1:
        guardarLocal_ = True
        hashGoogle()  # Realiza conexion con Google Drive
        ObtenerSonidos(Inicial_pNXML_, Final_pNXML_, ventana_Tiempo_, datos[0][1], Sin_Background_,
                       rutaDatosXML_, ruta_resultados, rutaDatosSounds_, Solo_Background_, datos[0][0],
                       window_length_stft_,
                       Step_size_stft_, guardarLocal_)

    elif opcion == 2:
        guardarLocal_ = False
        xmls = []
        for i in range(24):
            xmls.append([i + 1, i + 1])
        for i in datos:
            for j in xmls:
                ObtenerSonidos(j[0], j[1], ventana_Tiempo_, i[1], Sin_Background_,
                               rutaDatosXML_, ruta_resultados, rutaDatosSounds_, Solo_Background_, i[0],
                               window_length_stft_, Step_size_stft_, guardarLocal_)

    elif opcion == 3:
        hashGoogle()  # Realiza conexion con Google Drive
        nombre = "22050_4_mfcc"
        nombreModelo = 'Modelo_' + nombre + '.json'
        ruta = './drive/modelos/'
        guardarLocalmente(nombreModelo, ruta)
        nombreModelo = 'Pesos_Modelo_' + nombre + '.h5'
        ruta = './drive/modelos/pesos/'
        guardarLocalmente(nombreModelo, ruta)

    elif opcion == 4:
        mfcc = True
        esp = True
        nombre = titulo

        if mfcc and esp:
            modelo = modelo_doble;
            nombre += '_mfcc_y_spectrogram'
        elif mfcc:
            # modelo = modelo_mfcc;
            nombre += '_mfcc'
        elif esp:
            # modelo = modelo_esp;
            nombre += '_spectrogram'

        rutaModelo = "drive/My Drive/Proyecto Especial/Modelos/modelo_" + nombre + ".json"
        rutaPesos = "drive/My Drive/Proyecto Especial/Modelos/pesos_modelo_" + nombre + '.h5'
        rutaDiagrama = "drive/My Drive/Proyecto Especial/Modelos/diagrama_modelo_" + nombre + ".png"

        modelo = guardarModelo(modelo, rutaModelo, rutaPesos)
        print("Se Guardo, Suerte :) ")

    elif opcion == 5:
        pass

    elif opcion == 6:
        folds = crear_folds(numero_audios, numero_folds)
        sample_rate = datos[0][0]
        salto = datos[0][1]
        conf_matrix_array = []
        notify = Notify()
        for i in range(numero_folds):

            lista_y, lista_x2, lista_y_validacion, lista_x2_validacion, lista_x2_mfcc, lista_x2_esp, lista_x2_esp_validacion, lista_x2_mfcc_validacion = listas(
                folds, i, ruta_resultados, salto, ventana_tiempo_string,
                sample_rate, Sin_Background_, Solo_Background_,
                esp, mfcc, numero_audios)

            y_train = join('y_', lista_y)
            y_test = join('y_', lista_y_validacion)

            if esp and mfcc:
                x2_esp = join('x2_', lista_x2_esp)
                x2_mfcc = join('x2_', lista_x2_mfcc)
                x2_esp_validacion = join('x2_', lista_x2_esp_validacion)
                x2_mfcc_validacion = join('x2_', lista_x2_mfcc_validacion)
                x2_esp, x2_mfcc, y_train, pesosClases = over_under_pesos(under_over_pesos, esp, mfcc, x2_esp, x2_mfcc,
                                                                         y_train)

                x2_esp, x2_esp_validacion = reshape_data(x2_esp, x2_esp_validacion)
                x2_mfcc, x2_mfcc_validacion = reshape_data(x2_mfcc, x2_mfcc_validacion)

                numero_datos, alto_1_1, ancho_1_2, uno = x2_esp.shape
                numero_datos, alto_2_1, ancho_2_2, uno = x2_mfcc.shape

                modelo_doble = crearModelo_2D_Doble(lr, alpha, numFiltros, tamFiltros,
                                                    tamPooling, numNeuronas, optimizer,
                                                    T_entrada_1=alto_2_1, T_entrada_2=ancho_2_2,
                                                    T_entrada_3=alto_1_1, T_entrada_4=ancho_1_2)
                if i == 0:
                    modelo_doble.summary()
                    plot_model(modelo_doble, to_file='./drive/modelos/diagrama/model.png')

                modelo_doble, r1, r2 = entrenar(epocas, batch_size, modelo_doble, x2_mfcc,
                                                y_train, x2_mfcc_validacion, y_test,
                                                pesosClases, str(lr),
                                                esp, mfcc, x2_esp, x2_esp_validacion)

                y_prob = modelo_doble.predict([x2_esp_validacion, x2_mfcc_validacion])
                y_pred = y_prob.argmax(axis=-1)

                confusion_matrix_ = confusion_matrix(y_test, y_pred)
                confusion_matrix_ = confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]
                conf_matrix_array.append([confusion_matrix_])

                notify.send('va en el fold ' + str(i) + ' Train: %.3f, Test: %.3f' % (r1, r2))

        resultados(conf_matrix_array, ' resultado')

        notify.send('Revisa we, ya se acabo')
        # run notify_run register to subscribe and receive notifications

    elif opcion == 7:
        pbounds = {'lr': (0.001, 0.1),
                   'bs': (100, 500),
                   'dense1': (10, 50),
                   'dense2': (10, 50),
                   'dense3': (10, 50)
                   }

        optimizer = BayesianOptimization(
            f=evaluate_network,
            pbounds=pbounds,
            verbose=1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        start_time = time.time()
        optimizer.maximize(init_points=10, n_iter=100, )
        time_took = time.time() - start_time

        print(optimizer.max)
        print("Total runtime: " + str(time_took))
    else:
        break
