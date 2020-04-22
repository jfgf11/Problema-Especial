from metodos import *
import numpy as np
import csv

while True:
    print('Menu para seleccionar la accion a ejecutar')
    print('-------------------------------------------')
    print('1. Descargar Audios y xml de google drive.')
    print('2. Sacar x2 y y')
    print('3. Descargar modelo de Drive')
    print('4. Unir y')
    print('5. Unir x2')
    print('6. Crear y Entrenar RED 2D')
    print('7. Mostrar Matriz de confusion')
    print('8. Cross validation')
    print('9. Salir')
    print('-------------------------------------------')
    print('Porfavor seleccione una opción')
    print('CUIDADO: algunas de estas operaciones consumen bastante '
        'memoria y pueden trabar su computador, antes de correr algo '
        'este seguro de que cuenta con la memoria suficiente !!!')

    opcion=int(input())

    while not (0 < opcion < 10):
        print('Porfavor seleccione una opción entre 1 y 8')
        opcion=int(input())

    print('Usted selecciono la opción: ',opcion)

    ''''
    Parametros 
    '''

    if True: # parametros
        metodo_ = 'cross_validation'
        ventana_Tiempo_ = 0.700  # La ventana de tiempo de cada muestra (XX_ms)
        salto_de_ventana_ = 4  # Corrimiento en tiempo (XXms/4)
        sample_rate_ = 22050  # Tasa remuestreo
        Inicial_pNXML_ = 1  # Número inicial de archivos XML utilizados
        Final_pNXML_ = 15  # Número final de archivos XML utilizados
        Inicial_pNXML_test = 16  # Número inicial de archivos XML utilizados
        Final_pNXML_test = 18  # Número final de archivos XML utilizados
        Inicial_pNXML_validacion = 19  # Número inicial de archivos XML utilizados
        Final_pNXML_validacion = 21  # Número final de archivos XML utilizados
        Espectograma_ = True  # Calcular el espectograma
        MFCC_ = True  # Calcular el MFCC
        it=''

        window_length_stft_ = 0.025  # Ventana de tiempo para la short-Time Fourier Transform
        ventana_Tiempo_string='690'

        # Si se está obteniendo el espectogramo, el valor de la ventana no puede ser menor a 0.025s
        # Si se está obteniendo el MFCC, el valor de la ventana no puede ser menor a 0.03125s

        Step_size_stft_ = 0.010  # Saltos el en tiempo para la transformada de Fourier, fíjenlo menor a la ventana stft, si quieren pueden aumentar
        Sin_Background_ = False  # True: no se obtienen datos de background; False: No se obtienen datos de background  NO MOVER
        rutaDatosXML_ = "./drive/xml/"  # Ruta para encontrar archivos xml  NO MOVER
        rutaDatosSounds_ = "./drive/sounds/"  # Ruta para encontrar Audios  NO MOVER
        ruta_resultados = './drive/datos_procesados/'
        Solo_Background_ = False  # Solo obtener datos de background  NO MOVER
        guardarLocal_ = False

        if Espectograma_:
            Esp_o_Mfcc = "_espectrogram"
        elif MFCC_:
            Esp_o_Mfcc = "_MFCC"
        else:
            Esp_o_Mfcc = ""

        titulo = str(sample_rate_) + '_'+str(salto_de_ventana_)

    if opcion==1:
        guardarLocal_ = True
        hashGoogle() # Realiza conexion con Google Drive
        ObtenerSonidos(Inicial_pNXML_, Final_pNXML_,ventana_Tiempo_, salto_de_ventana_, Sin_Background_,
            rutaDatosXML_, ruta_resultados, rutaDatosSounds_, Solo_Background_, sample_rate_, window_length_stft_,
            Step_size_stft_, guardarLocal_)

    elif opcion==2:
        lista_de_Listas = [[22050, 4]]
        xmls = [[1,15],[16,18],[19,21]]
        for i in lista_de_Listas:
            for j in xmls:
                ObtenerSonidos( j[0], j[1], ventana_Tiempo_, i[1], Sin_Background_,
                           rutaDatosXML_, ruta_resultados, rutaDatosSounds_, Solo_Background_, i[0],
                           window_length_stft_, Step_size_stft_, guardarLocal_)

    elif opcion==3:
        hashGoogle()  # Realiza conexion con Google Drive
        print('Ingrese el nombre de quien realizó el modelo.')
        nombre=input()
        print('Ingrese la frecuencia de muestreo.')
        frec=input()
        print('MFCC o spectropgram?')
        mfcc = input()
        nombreModelo='Modelo_' + nombre + '_' + frec + '_' + mfcc +'.json'
        ruta = './drive/modelos/'
        guardarLocalmente(nombreModelo,ruta)
        nombreModelo = 'Pesos_Modelo_' + nombre + '_' + frec + '_' + mfcc + '.h5'
        ruta = './drive/modelos/pesos/'
        guardarLocalmente(nombreModelo, ruta)

    elif opcion==4:
        pass

    elif opcion==5:
        pass

    elif opcion==6:
        red_2d_cross_validation(ruta_resultados, ventana_Tiempo_, sample_rate_, 'oli', Sin_Background_, Solo_Background_,
                                MFCC_, Espectograma_)

    elif opcion==7:
        parametros = [[22050, 4],[22050, 4]]
        under_over_pesos = ['oversampling']
        f = open('drive/csv/resultados_cm.csv', 'w', newline='')
        with f:

            writer = csv.writer(f)
            for l in range(5): # iteraciones
                rows=[]
                for i in range(len(parametros)):
                    rows.append([])
                for i in parametros:
                    for j in under_over_pesos:

                        # --------------------------------------------------------------
                        # ---------------------- Extraer datos--------------------------
                        # --------------------------------------------------------------

                        titulo = str(i[0]) + '_' + str(i[1])

                        x2_esp,x2_mfcc,y_train = extrar_datos(Inicial_pNXML_, Final_pNXML_, Espectograma_, MFCC_, ruta_resultados,
                                    ventana_Tiempo_string, i[0], Sin_Background_, Solo_Background_, i[1])

                        x2_esp_test, x2_mfcc_test, y_test = extrar_datos(Inicial_pNXML_test, Final_pNXML_test, Espectograma_,
                                    MFCC_, ruta_resultados, ventana_Tiempo_string, i[0], Sin_Background_, Solo_Background_, i[1])

                        x2_esp_validacion, x2_mfcc_validacion, y_validacion = extrar_datos(Inicial_pNXML_validacion, Final_pNXML_validacion, Espectograma_,
                                    MFCC_, ruta_resultados,ventana_Tiempo_string, i[0], Sin_Background_, Solo_Background_, i[1])

                        # --------------------------------------------------------------
                        # ---------------------- Parametros Red-------------------------
                        # --------------------------------------------------------------

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

                        epocas = 100
                        batch_size = 5000

                        # --------------------------------------------------------------
                        # ---------------------- Entrenamiento Red----------------------
                        # --------------------------------------------------------------

                        if j == 'undersampling':
                            x2_esp, x2_mfcc, y_train_esp, y_train_mfcc = random_under_sampling(Espectograma_, MFCC_,x2_esp,x2_mfcc,y_train)
                            pesosClases = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=y_train_mfcc)
                        elif j == 'oversampling':
                            x2_esp, x2_mfcc, y_train_esp, y_train_mfcc = random_over_sampling(Espectograma_, MFCC_,x2_esp,x2_mfcc,y_train)
                            pesosClases = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=y_train_mfcc)
                        elif j =='pesos':
                            pesosClases = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=y_train)
                            y_train_esp = y_train
                            y_train_mfcc = y_train

                        x2_esp, x2_esp_test = reshape_data(x2_esp, x2_esp_test)
                        numero_datos, uno, alto, ancho = x2_esp.shape
                        modelo_esp = crearModelo2D(tasa, alpha, numFiltros, tamFiltros, tamPooling, numNeuronas, optimizer,
                                                T_entrada_1=alto, T_entrada_2=ancho)

                        modelo_esp=entrenar(epocas, batch_size, modelo_esp, x2_esp, y_train_esp, x2_esp_test, y_test, pesosClases, titulo + ' entrenamiento')


                        #x2_mfcc, x2_mfcc_test = reshape_data(x2_mfcc, x2_mfcc_test)
                        #numero_datos, uno, alto, ancho = x2_mfcc.shape
                        #modelo_mfcc = crearModelo2D(tasa, alpha, numFiltros, tamFiltros, tamPooling, numNeuronas, optimizer,
                                                    #T_entrada_1=alto, T_entrada_2=ancho)

                        #modelo_mfcc=entrenar(epocas, batch_size, modelo_mfcc, x2_mfcc, y_train_mfcc, x2_mfcc_test, y_test, pesosClases, titulo+ ' entrenamiento')

                        # --------------------------------------------------------------
                        # ------------- Validación y guardar resultados-----------------
                        # --------------------------------------------------------------

                        #Numero_Datos, alto, ancho = x2_mfcc_validacion.shape
                        #x2_mfcc_validacion = np.reshape(x2_mfcc_validacion, (-1, 1, alto, ancho), 'F')
                        #cm=graficarMatrizConfusion(y_validacion, modelo_mfcc.predict_classes(x2_mfcc_validacion),
                        #                        titulo + ' mfcc ' + j + it)

                        #for h in range(len(cm)):
                        #    for k in cm[h]:
                        #        rows[h].append(k)

                        #for row in rows:
                            #row.append('')

                        Numero_Datos, alto, ancho = x2_esp_validacion.shape
                        x2_esp_validacion = np.reshape(x2_esp_validacion, (-1, 1, alto, ancho), 'F')
                        cm=graficarMatrizConfusion(y_validacion, modelo_esp.predict_classes(x2_esp_validacion), titulo + ' espoctrogram ' + j + it)

                        #for h in range(len(cm)):
                        #    for k in cm[h]:
                        #        rows[h].append(k)

                        #for row in rows:
                        #    row.append('')

                #writer.writerow([''])
                #for row in rows:
                #    writer.writerow(row)


    elif opcion==8:
        red_2d_cross_validation(ruta_resultados, ventana_Tiempo_, sample_rate_, Sin_Background_, Solo_Background_,
                                MFCC_, Espectograma_)
    elif opcion==9:
        break
    else:
        break

print("Se acabo")