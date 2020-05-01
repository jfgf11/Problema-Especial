from typing import List

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.utils import plot_model

import keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score

K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

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

def guardarModelo(pModelo, pRutaModelo, pRutaPesos):
    modelo_json = pModelo.to_json()

    with open(pRutaModelo, "w") as archivo_json:
        archivo_json.write(modelo_json)

    pModelo.save_weights(pRutaPesos)

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

def crear_folds(numero_audios, numero_folds):
    base_audios = int(numero_audios / numero_folds)
    numero_bases_con_un_audio_mas = numero_audios % numero_folds
    audios = [0]
    cuenta = 0
    for j in range(numero_folds):
        if j < numero_bases_con_un_audio_mas:
            cuenta += base_audios + 1
            audios.append(cuenta)
        else:
            cuenta += base_audios
            audios.append(cuenta)
    return audios

def reshape_data(x_train, x_test):
    numero_Datos, alto, ancho = x_train.shape
    x_train = np.reshape(x_train, (-1, alto, ancho, 1), 'F')
    x_test = np.reshape(x_test, (-1, alto, ancho, 1), 'F')
    return x_train, x_test

def random_over_sampling(x2, y_train):
    numero_datos, alto, ancho = x2.shape
    x2 = np.reshape(x2, (-1, alto * ancho), 'F')

    ros = RandomOverSampler(sampling_strategy='not majority', random_state=0)
    x2, y_train = ros.fit_resample(x2, y_train)

    x2 = np.reshape(x2, (-1, alto, ancho), 'F')

    return x2, y_train

def obtener_datos(folds, i, numero_audios):

    ventana_tiempo_string = "450"
    esp = True
    mfcc = True
    datos = [22050, 4]
    ruta_resultados = './drive/datos_procesados/'
    Sin_Background_ = False
    Solo_Background_ = False

    lista_y, lista_x2, lista_y_validacion, lista_x2_validacion, lista_x2_mfcc, lista_x2_esp, lista_x2_esp_validacion, lista_x2_mfcc_validacion = listas(
        folds, i, ruta_resultados, datos[1], ventana_tiempo_string, datos[0], Sin_Background_, Solo_Background_,
        esp, mfcc, numero_audios)

    y_train = join('y_', lista_y)
    y_test = join('y_', lista_y_validacion)
    x2_esp = join('x2_', lista_x2_esp)
    x2_mfcc = join('x2_', lista_x2_mfcc)
    x2_esp_test = join('x2_', lista_x2_esp_validacion)
    x2_mfcc_test= join('x2_', lista_x2_mfcc_validacion)

    x2_esp,  y_train_esp = random_over_sampling(x2_esp,  y_train)
    x2_mfcc, y_train_mfcc = random_over_sampling(x2_mfcc,  y_train)

    x2_esp, x2_esp_test =reshape_data(x2_esp, x2_esp_test)
    x2_mfcc, x2_mfcc_test =reshape_data(x2_mfcc, x2_mfcc_test)

    y_train = np.reshape(y_train_esp, (y_train_esp.shape[0], 1), 'F')
    y_test = np.reshape(y_test, (y_test.shape[0], 1), 'F')

    return [x2_esp, x2_mfcc], [x2_esp_test, x2_mfcc_test], y_train, y_test

def mejor_modelo2D_doble(esp_shape, mfcc_shape):

    # --------------------------------------------------------------
    # ------------------ CNN Spec ---------------------------------
    # --------------------------------------------------------------

    capaEntrada_1 = Input(esp_shape)

    # Capa 1
    X = Conv2D(kernel_size=2, filters=32, padding='same', kernel_initializer=glorot_uniform(seed=0))(capaEntrada_1)
    #X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    #X = AveragePooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 2
    X = Conv2D(kernel_size=3, filters=32, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(10, 10), strides=[1, 1], padding='same')(X)

    # Capa 3
    # X = Conv2D(pNumFiltros[2], int(pTamFiltros[2],), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D(int(pTamPooling[2]), padding='same')(X)

    flatten_1 = Flatten()(X)

    # --------------------------------------------------------------
    # ------------------ CNN MFCC ----------------------------------
    # --------------------------------------------------------------
    capaEntrada_2 = Input(mfcc_shape)

    # Capa 1
    #X = ZeroPadding2D(padding=(0, 1))(capaEntrada_2)
    X = Conv2D(kernel_size=2, filters=32, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    #X = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 2
    X = Conv2D(kernel_size=3, filters=32, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(10, 10), strides=[1, 1], padding='same')(X)

    # Capa 3
    #X = Conv2D(kernel_size=3, filters=64, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = Activation('relu')(X)
    #X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

    # Capa 4
    # X = Conv2D(kernel_size=3, filters=64, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = Activation('relu')(X)

    flatten_2 = Flatten()(X)
    capas = concatenate([flatten_1, flatten_2])

    # red neuronal

    capas = Dropout(0.5)(capas)
    capas = Dense(64, activation='relu')(capas)
    capas = Dense(32, activation='relu')(capas)
    #capas = Dense(20, activation='relu')(capas)

    capaSalida = Dense(4, activation='softmax')(capas)

    modelo = Model(inputs=[capaEntrada_1, capaEntrada_2], outputs=capaSalida)

    return modelo

def crearModelo_2D_Doble(esp_shape, mfcc_shape):

    # --------------------------------------------------------------
    # ------------------ CNN Spec ---------------------------------
    # --------------------------------------------------------------

    capaEntrada_1 = Input(esp_shape)

    # Capa 1
    X = Conv2D(kernel_size=5, filters=6, kernel_initializer=glorot_uniform(seed=0))(capaEntrada_1)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 2
    X = Conv2D(kernel_size=5, filters=16, kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 3
    # X = Conv2D(pNumFiltros[2], int(pTamFiltros[2],), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D(int(pTamPooling[2]), padding='same')(X)

    flatten_2 = Flatten()(X)

    # --------------------------------------------------------------
    # ------------------ CNN MFCC ----------------------------------
    # --------------------------------------------------------------
    capaEntrada_2 = Input(mfcc_shape)

    # Capa 1
    X = ZeroPadding2D(padding=(0, 1))(capaEntrada_2)
    X = Conv2D(kernel_size=3, filters=16, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 2
    X = Conv2D(kernel_size=3, filters=32, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(X)

    # Capa 3
    #X = Conv2D(kernel_size=3, filters=64, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = Activation('relu')(X)
    #X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

    # Capa 4
    # X = Conv2D(kernel_size=3, filters=64, padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = Activation('relu')(X)

    flatten_1 = Flatten()(X)
    capas = concatenate([flatten_1, flatten_2])

    # red neuronal

    # capas = Dropout(0.5)(capas)
    capas = Dense(10, activation='relu')(capas)
    capas = Dense(10, activation='relu')(capas)
    #capas = Dense(20, activation='relu')(capas)

    capaSalida = Dense(4, activation='softmax')(capas)

    modelo = Model(inputs=[capaEntrada_1, capaEntrada_2], outputs=capaSalida)

    return modelo

def plot_results(hist):

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
    # plt.plot(y_entrenamiento, label='F1 Entrenamiento')
    # plt.plot(y_validacion, label='F1 Validacion')
    # plt.xlabel('Epoca')
    # plt.ylabel('F1_Score')
    # plt.title("F1_score vs Epoca")
    # plt.legend()
    # plt.show()

def plot_f1_results(f1_train,f1_test):
    plt.plot(f1_train, label='F1 Entrenamiento')
    plt.plot(f1_test, label='F1 Validacion')
    plt.xlabel('Epoca')
    plt.ylabel('F1_Score')
    plt.title("F1_score vs Epoca")
    plt.legend()
    plt.show()

def resultados(conf_matrix_array):
    arr = np.mean(conf_matrix_array, axis=0)
    df_cm = pd.DataFrame(arr[0], index=[i for i in "0123"], columns=[i for i in "0123"])
    plot = plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True)
    plt.xlabel("Clase Prediccion")
    plt.ylabel("Clase Verdadera")
    plt.title("Matriz de Confusion")
    plt.show()
    plot.savefig('./drive/modelos/matriz/CMNN.png')


numero_audios = 24
numero_folds = 24
folds=crear_folds(numero_audios, numero_folds)  # type: List[int]

conf_matrix_array = []

for i in range(1):

    f1_train = []
    f1_test = []

    X_train, X_test, Y_train, Y_test = obtener_datos(folds, i, numero_audios)

    x_esp_train = X_train[0]
    x_mfcc_train = X_train[1]

    x_esp_test = X_test[0]
    x_mfcc_test = X_test[1]

    modelo = crearModelo_2D_Doble(x_esp_train.shape[1:],x_mfcc_train.shape[1:])

    # opt = tf.keras.optimizers.Adam(learning_rate=lr)
    modelo.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    for epoca in range(15):
        hist=modelo.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), epochs=1,batch_size=5, verbose=2)
        f1_train.append(f1_score(Y_train, (modelo.predict(X_train)).argmax(axis=-1), average='macro'))
        f1_test.append(f1_score(Y_test, (modelo.predict(X_test)).argmax(axis=-1), average='macro'))
    # plot_results(hist)
    plot_f1_results(f1_train,f1_test)

    # preds_test = modelo.evaluate(x=X_test, y=Y_test, verbose=0)
    # preds_train = modelo.evaluate(x=X_train, y=Y_train, verbose=0)
    # print('Train: %.3f, Test: %.3f' % (preds_test[1], preds_train[1]))

    y_prob=modelo.predict(X_test)
    y_pred = y_prob.argmax(axis=-1)
    # graficarMatrizConfusion(Y_test, y_pred)
    # confusion_matrix = graficarMatrizConfusion(y_test, y_pred, titulo + j)

    confusion_matrix_ = confusion_matrix(Y_test, y_pred)
    confusion_matrix_ = confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]
    conf_matrix_array.append([confusion_matrix_])

    if i == 0:
        modelo.summary()
        plot_model(modelo, to_file='./drive/modelos/diagrama/model.png')
        print("number of training examples = " + str(x_esp_train.shape[0]))
        print("number of test examples = " + str(x_esp_test.shape[0]))
        print("X_train shape: ", x_esp_train.shape, x_mfcc_train.shape)
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: ", x_esp_test.shape, x_mfcc_test.shape)
        print("Y_test shape: " + str(Y_test.shape))
    else:
        print('-------------------------------------------')
        print('----------------- Fold ' + str(i+2)+ '-----------------')
        print('-------------------------------------------')

resultados(conf_matrix_array)

mfcc = True
esp = True

rutaModelo = "drive/modelos/modelo.json"
rutaPesos = "drive/modelos/pesos/pesos_modelo.h5"

modelo=guardarModelo(modelo, rutaModelo, rutaPesos)
print("Se Guardo, Suerte :) ")