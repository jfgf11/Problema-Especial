from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import psutil
import tensorflow.compat.v1 as tf
# Tensor Flow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model
# Recopilacion de datos
import xml.dom.minidom
import numpy as np
# Para el preprocesamiento
from sklearn.model_selection import train_test_split
import librosa
import progressbar
# Import libraries
import librosa.display
#Redes neuronales sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.utils.class_weight import compute_class_weight
import collections
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

tf.disable_v2_behavior()

''''
Metodos Google drive 
'''

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
    file_list = drive.ListFile({'q': "'"+str(d[root])+"' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        name=file1['title']
        d[name]=file1['id']
        if '.' not in file1['title']:
            agregar_hash(name)

def hashGoogle():
    global drive
    global d

    drive = autorizacionGoogle()
    d={}

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

    if str_variable== 'y_':
        ruta = carpeta + str_variable+ 'salto_' + str(salto) + '_' + str(Inicial_pNXML) + "-" + str(Final_pNXML) + \
               "_Audios_" + ventana_Tiempo + "ms_" + sample_rate_String + Back + Esp_o_Mfcc
    else:
        ruta = carpeta + str_variable + 'salto_' + str(salto) + '_' +str(Inicial_pNXML) + "-" + str(Final_pNXML) + "_Audios_" + ventana_Tiempo \
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

    ventana_Tiempo_str='690'
    if not guardarLocal:
        NMV = round(ventana_Tiempo * sample_rate)  # Numero de muestras por ventana
        NMV_advance = round(NMV / salto_de_ventana)  # Numero de muestras por las cuales se avanza
        # 190 representa que cada audio dura aproximadamente 3 minutos y 10 segundos
        numero_datos=int(((190 / ventana_Tiempo) * salto_de_ventana) * (Final_pNXML - Inicial_pNXML + 1))
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
                start = doc.getElementsByTagName("STARTSECOND")  # Vector que contiene el tiempo en segundos de inicio de todos los eventos
                finish = doc.getElementsByTagName("ENDSECOND")  # Vector que contiene el tiempo en segundos de finalizacion de todos los eventos
                ID = doc.getElementsByTagName("CLASS_ID")  # Vector que contiene la etiqueta de cada uno de los eventos
                events = doc.getElementsByTagName("events")  # Indica informacion de todos los eventos en un archivo xml (tamaño)
                a, b, c, d = (events[0].attributes["size"].value)  # Se obtiene el numero de eventos en un audio
                nEventos = int(c + d)  # numero de eventos en un audio

            archivo_audio = '00' + h + str(i) + '.wav' # cada xml solo tiene un audio asociado.
            if guardarLocal:
                guardarLocalmente(archivo_audio, rutaDatosSounds)
            else:
                frameData, fs = librosa.load(rutaDatosSounds + archivo_audio, sr=sample_rate, res_type='kaiser_fast')  # Audio seleccionado
                datos_x = (librosa.util.frame(frameData, frame_length=NMV, hop_length=NMV_advance)).T  # Reorganiza los datos dándole saltos de tiempo de NMV_advance y el número de muestras por ventana NMV
                datos_y = np.zeros(len(frameData))  # Etiquetas de cada uno de los datos, los datos no asignados serán 0 y corresponderan a sonido ambiente

                for j in range(0, nEventos):  # Se recorre el numero de eventos para cada xml
                    startFrame = float(str(start[j].firstChild.data)) * fs  # Posicion inicial de evento con respecto a frameData
                    endFrame = float(str(finish[j].firstChild.data)) * fs  # Posicion final de evento con respecto a frameData
                    label = ID[j].firstChild.data  # etiqueta del evento
                    datos_y[round(startFrame):round(endFrame)] = int(label)  # Se asigna la etiqueta a cada uno de los datos recopilados

                datos_y = (((stats.mode(librosa.util.frame(datos_y, frame_length=NMV, hop_length=NMV_advance)))[0]).T)  # Con esto se asigna la etiqueta a datos desplazados en el tiempo
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
        ruta = dar_ruta(ruta_datos, 'x2_', salto_de_ventana, Inicial_pNXML, Final_pNXML, ventana_Tiempo_str, sample_rate,
            Sin_Background, Solo_Background, True, False)
        np.save(ruta, x2)

        x2 = None

        x2 = Crear_Datos_MFCC_o_Espectogram(datos_x_totales, fs, window_length_stft,
                                            Step_size_stft, True, False, ventana_Tiempo)
        ruta = dar_ruta(ruta_datos, 'x2_', salto_de_ventana, Inicial_pNXML, Final_pNXML, ventana_Tiempo_str, sample_rate,
                        Sin_Background, Solo_Background, False, True)
        np.save(ruta, x2)

        longitud_actual = 0
        datos_x_totales = None
        x2 = None

'''''
Guardar y cargar modelo.
'''

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

'''''
Extraccion de datos juardados y union de ellos.
Preprocesamiento de los datos 
'''

def join(variable, lista):

    ans = None
    if variable == 'y_':
        y_def = None
        anterior = 0
        for i in range(len(lista)):
            y = np.load(lista[i])

            if i == 0:
                y_def = np.zeros(y.size * 2* (len(lista)))

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
        ans=x2_def
        print(x2_def.shape)
        x2 = None
        x2_def = None

    return ans

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


def red_2d_cross_validation(ruta_resultados, ventana_Tiempo, sample_rate, nombre, Sin_Background, Solo_Background,
                            MFCC, Espectogram):
    numFiltros = np.array([12, 20, 20, 12, 10, 512])
    tamFiltros = np.array([3, 3, 3, 3, 3, 5])
    tamPooling = np.array([4, 2, 3, 3, 3, 3])
    numNeuronas = np.array([10, 10, 10, 16])
    optimizer = "rmsprop"
    tasa = 0.1
    alpha = 0.01

    numero_folds = 5
    numero_audios = 20
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
    for i in range(numero_folds):
        lista_y = []
        lista_x2 = []
        lista_y_validacion = []
        lista_x2_validacion = []
        for j in range(1, audios[i] + 1):
            #ruta = dar_ruta(ruta_resultados, 'y_', j, j, ventana_Tiempo, sample_rate, nombre,
            #                Sin_Background, Solo_Background, Espectogram, MFCC)
            #lista_y.append(ruta + '.npy')
            #ruta = dar_ruta(ruta_resultados, 'x2_', j, j, ventana_Tiempo, sample_rate, nombre,
            #                Sin_Background, Solo_Background, Espectogram, MFCC)
            #lista_x2.append(ruta + '.npy')
            lista_x2.append(j)

        for j in range(audios[i] + 1, audios[i + 1] + 1):
            #ruta = dar_ruta(ruta_resultados, 'y_', j, j, ventana_Tiempo, sample_rate, nombre,
            #                Sin_Background, Solo_Background, Espectogram, MFCC)
            #lista_y_validacion.append(ruta + '.npy')
            #ruta = dar_ruta(ruta_resultados, 'x2_', j, j, ventana_Tiempo, sample_rate, nombre,
            #                Sin_Background, Solo_Background, Espectogram, MFCC)
            #lista_x2_validacion.append(ruta + '.npy')
            lista_x2_validacion.append(j)

        for j in range(audios[i + 1] + 1, numero_audios + 1):
            #ruta = dar_ruta(ruta_resultados, 'y_', j, j, ventana_Tiempo, sample_rate, nombre,
            #                Sin_Background, Solo_Background, Espectogram, MFCC)
            #lista_y.append(ruta + '.npy')
            #ruta = dar_ruta(ruta_resultados, 'x2_', j, j, ventana_Tiempo, sample_rate, nombre,
            #                Sin_Background, Solo_Background, Espectogram, MFCC)
            #lista_x2.append(ruta + '.npy')
            lista_x2.append(j)

        print(lista_x2, lista_x2_validacion)
        #y_train = join('y_', lista_y)
        #x_train_2 = join('x2_', lista_x2)
        #y_test = join('y_', lista_y_validacion)
        #x_test_2 = join('x2_', lista_x2_validacion)

        #numero_datos, alto_2, ancho_2 = x_train_2.shape
        #modelo = crearModelo2D(tasa, alpha, numFiltros, tamFiltros, tamPooling, numNeuronas, optimizer,
        #                        T_entrada_1=alto_2, T_entrada_2=ancho_2)
        #x_train_2 = np.reshape(x_train_2, (-1, 1, alto_2, ancho_2), 'F')
        #x_test_2 = np.reshape(x_test_2, (-1, 1, alto_2, ancho_2), 'F')

        #pesos = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=y_train)
        #titulo='validacion con ' + str(audios[i]) + '-' + str(audios[i+1])
        #entrenar_cross(x_train_2, y_train, x_test_2, y_test,pesos,modelo,titulo)

def entrenar_cross(x_train_2, y_train, x_test_2, y_test,pesos,modelo,titulo):
    epocas = 10
    batchSize = 5000
    modelo.compile(loss='sparse_categorical_crossentropy', optimizer = "rmsprop", metrics = ['sparse_categorical_accuracy'])
    for i in range(0, 1):
        hist = modelo.fit(x_train_2, y_train, validation_data=(x_test_2, y_test), epochs=epocas,
                          batch_size=batchSize, class_weight=pesos)

    graficarMatrizConfusion(y_test, modelo.predict_classes(x_test_2), titulo)

def entrenar(epocas, batch_size, modelo, x_train_2, y_train, x_test_2, y_test, pesos,titulo):

    modelo.compile(loss='sparse_categorical_crossentropy', optimizer = "rmsprop", metrics = ['sparse_categorical_accuracy'])
    if batch_size!=-1:
        for i in range(0, 1):
            hist = modelo.fit(x_train_2, y_train, validation_data=(x_test_2, y_test), epochs=epocas,
                          batch_size=batch_size, class_weight=pesos)
    else :
        for i in range(0, 1):
            hist = modelo.fit(x_train_2, y_train, validation_data=(x_test_2, y_test), epochs=epocas, class_weight=pesos)

    #graficarMatrizConfusion(y_test, modelo.predict_classes(x_test_2), titulo)

    return modelo

def matriz_confusion(modelo, titulo, x2, y):
    modelo.compile(loss='sparse_categorical_crossentropy', optimizer = "rmsprop", metrics = ['sparse_categorical_accuracy'])
    Numero_Datos, alto_2, ancho_2 = x2.shape
    x2 = np.reshape(x2, (-1, 1, alto_2, ancho_2), 'F')
    y_pred=modelo.predict_classes(x2)
    graficarMatrizConfusion(y,y_pred,titulo)

'''''
Creacion y entrenamiento modelo red 2D con validacion cross validation
'''

''''
Visualización de resultados
'''

def graficarMatrizConfusion(y_true, y_pred,titulo):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 8))

    ax = sns.heatmap(cm, annot=True, cbar=False);

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.xlabel("Clase Prediccion")
    plt.ylabel("Clase Verdadera")
    plt.title("Matriz de Confusion "+titulo)
    plt.savefig('./drive/modelos/matriz/' + titulo)
    plt.show()
    return cm

def extrar_datos(Inicial_pNXML, Final_pNXML,Espectrogram, MFCC,carpeta,
             ventana_Tiempo_string = "690", sample_rate = 22050, Sin_Background = False, Solo_Background =False,
             salto=4):
    x_spect = None
    x_MFCC = None
    y = None
    if Espectrogram:
        ruta = dar_ruta(carpeta, 'x2_', salto, Inicial_pNXML, Final_pNXML,
              ventana_Tiempo_string, sample_rate, Sin_Background, Solo_Background, True, False)

        x_spect = np.load(ruta+".npy")
    if MFCC:
        ruta = dar_ruta(carpeta, 'x2_', salto, Inicial_pNXML, Final_pNXML,
                    ventana_Tiempo_string, sample_rate, Sin_Background, Solo_Background, False, True)

        x_MFCC = np.load(ruta+".npy")
    ruta = dar_ruta(carpeta, 'y_', salto, Inicial_pNXML, Final_pNXML,
              ventana_Tiempo_string, sample_rate, Sin_Background, Solo_Background, True, False)

    y = np.load(ruta+".npy")
    return x_spect,x_MFCC,y

def reshape_data(x_train, x_test):
    numero_Datos, alto, ancho=x_train.shape
    x_train = np.reshape(x_train, (-1,1, alto, ancho), 'F')
    x_test = np.reshape(x_test, (-1,1, alto, ancho), 'F')
    return x_train, x_test

def random_under_sampling(mfcc,esp,x2_esp,x2_mfcc,y_train):
    if esp:
        numero_datos, alto, ancho = x2_esp.shape
        x2_esp = np.reshape(x2_esp, (-1, alto * ancho), 'F')

        rus = RandomUnderSampler(random_state=42)
        x2_esp, y_train_esp = rus.fit_resample(x2_esp, y_train)

        x2_esp = np.reshape(x2_esp, (-1, alto, ancho), 'F')

    if mfcc:
        numero_datos, alto, ancho = x2_mfcc.shape
        x2_mfcc = np.reshape(x2_mfcc, (-1, alto*ancho), 'F')

        rus = RandomUnderSampler(random_state=42)
        x2_mfcc, y_train_mfcc = rus.fit_resample(x2_mfcc, y_train)

        x2_mfcc = np.reshape(x2_mfcc, (-1, alto, ancho), 'F')

    return x2_esp, x2_mfcc, y_train_esp, y_train_mfcc

def random_over_sampling(mfcc,esp,x2_esp,x2_mfcc,y_train):
    if esp:
        numero_datos, alto, ancho = x2_esp.shape
        x2_esp = np.reshape(x2_esp, (-1, alto * ancho), 'F')

        ros = RandomOverSampler(sampling_strategy = 'not majority', random_state = 0)
        x2_esp, y_train_esp = ros.fit_resample(x2_esp, y_train)

        x2_esp = np.reshape(x2_esp, (-1, alto, ancho), 'F')

    if mfcc:
        numero_datos, alto, ancho = x2_mfcc.shape
        x2_mfcc = np.reshape(x2_mfcc, (-1, alto*ancho), 'F')

        ros = RandomOverSampler(sampling_strategy = 'not majority', random_state = 0)
        x2_mfcc, y_train_mfcc = ros.fit_resample(x2_mfcc, y_train)

        x2_mfcc = np.reshape(x2_mfcc, (-1, alto, ancho), 'F')

    return x2_esp, x2_mfcc, y_train_esp, y_train_mfcc

def train_test(x2_esp,x2_mfcc,y_train):
    x2_esp, x2_esp_test, y_train_a, y_test = train_test_split(x2_esp, y_train , random_state = 0, test_size=0.04)
    x2_mfcc, x2_mfcc_test, y_train_a, y_test = train_test_split(x2_mfcc, y_train , random_state = 0, test_size=0.04)
    return x2_esp, x2_esp_test, x2_mfcc, x2_mfcc_test, y_train_a, y_test