from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import psutil

import tensorflow.compat.v1 as tf


# Recopilacion de datos
import xml.dom.minidom
import numpy as np

import librosa
import progressbar

# Import libraries
import librosa.display

#Redes neuronales sklearn
from scipy import stats
from sklearn.model_selection import train_test_split

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

    return z, fs

'''''
Extraccion de datos juardados y union de ellos.
Preprocesamiento de los datos 
'''

def join(variable,z,rutaDatosParciales,ruta_resultados,Features,Inicial_pNXML,Final_pNXML,Inicial_nAudios,Final_nAudios,
         ventana_Tiempo,sample_rate,nombre,Sin_Background,Solo_Background,):
    if variable=='x':
        print('mk no, el pc no da, no lo intentes we')
    elif variable=='y':

        y = []

        yTotal = []
        for i in range(z):
            ruta = dar_ruta(rutaDatosParciales, Features, 'y_' + str(i) + '_', Inicial_pNXML, Final_pNXML,
                Inicial_nAudios, Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
                False, False)
            y = np.load(ruta + '.npy')
            yTotal.concatenate(y)
        ruta = dar_ruta(ruta_resultados, Features, 'y_', Inicial_pNXML, Final_pNXML, Inicial_nAudios,
            Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
            False, False)
        np.save(ruta, y)
        y = None
        yTotal=None

    elif variable=='x2':

        x2 = []

        x2Total = []
        for i in range(z):
            ruta = dar_ruta(rutaDatosParciales, Features, 'x2_' + str(i) + '_', Inicial_pNXML, Final_pNXML,
                            Inicial_nAudios, Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background,
                            Solo_Background,
                            False, False)
            y = np.load(ruta + '.npy')
            x2Total.concatenate(y)
        ruta = dar_ruta(ruta_resultados, Features, 'x2_', Inicial_pNXML, Final_pNXML, Inicial_nAudios,
                        Final_nAudios, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
                        False, False)
        np.save(ruta, y)
        y = None

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
