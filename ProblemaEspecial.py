from metodos import *

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
    print('8. Modificar parametros')
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
        window_length_stft_ = 0.025  # Ventana de tiempo para la short-Time Fourier Transform
        Inicial_pNXML_ = 1  # Número inicial de archivos XML utilizados  NO MOVER, a menos de que se quiera obtener 45 a 55
        Final_pNXML_ = 10  # Número final de archivos XML utilizados  NO MOVER, a menos de que se quiera obtener 45 a 55
        Espectograma_ = True  # Calcular el espectograma
        MFCC_ = False  # Calcular el MFCC
        validacion=True
        nombre_ = ""  # PONER NOMBRE QUIEN REALIZA LA PRUEBA

        ventana_Tiempo_ = 0.100  # La ventana de tiempo de cada muestra (XX_ms)
        ventana_Tiempo_string='100'
        salto_de_ventana_ = 4  # Corrimiento en tiempo (XXms/4)
        sample_rate_ = 22050  # Tasa remuestreo
        # Si se está obteniendo el espectogramo, el valor de la ventana no puede ser menor a 0.025s
        # Si se está obteniendo el MFCC, el valor de la ventana no puede ser menor a 0.03125s

        Step_size_stft_ = 0.010  # Saltos el en tiempo para la transformada de Fourier, fíjenlo menor a la ventana stft, si quieren pueden aumentar
        Sin_Background_ = False  # True: no se obtienen datos de background; False: No se obtienen datos de background  NO MOVER
        Inicial_nAudios_ = 0  # Número inicial de audios que se obtendrán (esto no aplica para nuestros audios)  NO MOVER
        Final_nAudios_ = 7  # Número final de audios que se obtendrán (esto no aplica para nuestros audios)  NO MOVER
        rutaDatosXML_ = "./drive/xml/"  # Ruta para encontrar archivos xml  NO MOVER
        rutaDatosSounds_ = "./drive/sounds/"  # Ruta para encontrar Audios  NO MOVER
        ruta_resultados = './drive/datos_procesados/'
        ruta_datos_parciales_ = "./drive/parciales/"
        Solo_Background_ = False  # Solo obtener datos de background  NO MOVER
        guardarLocal_ = False

        if Espectograma_:
            Esp_o_Mfcc = "_spectrogram"
        elif MFCC_:
            Esp_o_Mfcc = "_MFCC"
        else:
            Esp_o_Mfcc = ""
        if nombre_ != "":
            nombre_ = '_' + nombre_

        titulo = str(Inicial_pNXML_) + "-" + str(Final_pNXML_)+nombre_ + '_' + str(sample_rate_) + Esp_o_Mfcc

    if opcion==1:
        guardarLocal_ = True
        hashGoogle() # Realiza conexion con Google Drive
        z = ObtenerSonidos(nombre=nombre_, Inicial_pNXML=Inicial_pNXML_, Final_pNXML=Final_pNXML_,
                           Inicial_nAudios=Inicial_nAudios_, Final_nAudios=Final_nAudios_, ventana_Tiempo=ventana_Tiempo_,
                           salto_de_ventana=salto_de_ventana_, Sin_Background=Sin_Background_,
                           rutaDatosXML=rutaDatosXML_, rutaDatosParciales=ruta_datos_parciales_,
                           rutaDatosSounds=rutaDatosSounds_,
                           Solo_Background=Solo_Background_, sample_rate=sample_rate_,
                           window_length_stft=window_length_stft_,
                           Step_size_stft=Step_size_stft_, guardarLocal=guardarLocal_, MFCC=MFCC_, Espectogram=Espectograma_)

    elif opcion==2:
        a=[26]
        for i in a:
            z = ObtenerSonidos(nombre=nombre_, Inicial_pNXML=i, Final_pNXML=i+4,
                               Inicial_nAudios=Inicial_nAudios_, Final_nAudios=Final_nAudios_, ventana_Tiempo=ventana_Tiempo_,
                               salto_de_ventana=salto_de_ventana_, Sin_Background=Sin_Background_,
                               rutaDatosXML=rutaDatosXML_, rutaDatosParciales=ruta_datos_parciales_,
                               rutaDatosSounds=rutaDatosSounds_,
                               Solo_Background=Solo_Background_, sample_rate=sample_rate_,
                               window_length_stft=window_length_stft_,
                               Step_size_stft=Step_size_stft_, guardarLocal=guardarLocal_, MFCC=MFCC_, Espectogram=Espectograma_)

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
        lista = []  # Lista en la que se guardaran las rutas de los archivos a unir

        Inicial_pNXML_ = 46
        Inicial_pNXML_guardar = Inicial_pNXML_  # Se guarda el xml desde donde inicia
        Final_pNXML_ = 50
        ruta = dar_ruta(ruta_datos_parciales_, 'y_', Inicial_pNXML_, Final_pNXML_,
                        Inicial_nAudios_, Final_nAudios_, ventana_Tiempo_string, sample_rate_, nombre_,
                        Sin_Background_, Solo_Background_, Espectograma_, MFCC_)
        lista.append(ruta + '.npy')

        # En teoria entre archivos a unir solo deberia de cambiar el xml inicial y final.
        Inicial_pNXML_ = 51
        Final_pNXML_ = 55
        Final_pNXML_guardar=Final_pNXML_
        ruta = dar_ruta(ruta_datos_parciales_, 'y_', Inicial_pNXML_, Final_pNXML_,
                        Inicial_nAudios_, Final_nAudios_, ventana_Tiempo_string, sample_rate_, nombre_,
                        Sin_Background_, Solo_Background_, Espectograma_, MFCC_)
        lista.append(ruta + '.npy')
        # Si quiere unir mas de 2 archivos solo copie y pegue lo de arriba

        # print(lista)
        join('y_', lista, ruta_resultados, Inicial_pNXML_guardar, Final_pNXML_guardar, Inicial_nAudios_,
             Final_nAudios_,
             ventana_Tiempo_string, sample_rate_, nombre_, Sin_Background_, Solo_Background_, MFCC_, Espectograma_)

    elif opcion==5:
        lista = []  # Lista en la que se guardaran las rutas de los archivos a unir

        Inicial_pNXML_ = 46
        Inicial_pNXML_guardar = Inicial_pNXML_  # Se guarda el xml desde donde inicia
        Final_pNXML_ = 50
        ruta = dar_ruta(ruta_datos_parciales_, 'x2_', Inicial_pNXML_, Final_pNXML_,
                        Inicial_nAudios_, Final_nAudios_, ventana_Tiempo_string, sample_rate_, nombre_,
                        Sin_Background_, Solo_Background_, Espectograma_, MFCC_)
        lista.append(ruta + '.npy')

        # En teoria entre archivos a unir solo deberia de cambiar el xml inicial y final.
        Inicial_pNXML_ = 51
        Final_pNXML_ = 55
        Final_pNXML_guardar = Final_pNXML_
        ruta = dar_ruta(ruta_datos_parciales_, 'x2_', Inicial_pNXML_, Final_pNXML_,
                        Inicial_nAudios_, Final_nAudios_, ventana_Tiempo_string, sample_rate_, nombre_,
                        Sin_Background_, Solo_Background_, Espectograma_, MFCC_)
        lista.append(ruta + '.npy')
        # Si quiere unir mas de 2 archivos solo copie y pegue lo de arriba

        # print(lista)
        join('x2_', lista, ruta_resultados, Inicial_pNXML_guardar, Final_pNXML_guardar, Inicial_nAudios_,
             Final_nAudios_,
             ventana_Tiempo_string, sample_rate_, nombre_, Sin_Background_, Solo_Background_, MFCC_, Espectograma_)

    elif opcion==6:
        entrenarRed(ruta_resultados, Inicial_pNXML_, Final_pNXML_, Inicial_nAudios_, Final_nAudios_,
                    ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_, MFCC_, Espectograma_,
                    titulo)

    elif opcion==7:
        rutaModelo = './drive/modelos/Modelo_' + titulo + '.json'
        rutaPesos = './drive/modelos/pesos/Pesos_Modelo_' + titulo + '.h5'
        Inicial_pNXML_validacion=46
        Final_pNXML_validacion=55
        if validacion:
            titulo=titulo+' validacion_' + Inicial_pNXML_validacion + '_' + Final_pNXML_validacion
            rutay = dar_ruta(ruta_resultados, 'y_', Inicial_pNXML_validacion, Final_pNXML_validacion, Inicial_nAudios_,
                             Final_nAudios_, ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_,
                             Espectograma_, MFCC_)
            rutax2 = dar_ruta(ruta_resultados, 'x2_', Inicial_pNXML_validacion, Final_pNXML_validacion, Inicial_nAudios_,
                              Final_nAudios_, ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_,
                              Espectograma_, MFCC_)
        else:
            rutay = dar_ruta(ruta_resultados, 'y_', Inicial_pNXML_, Final_pNXML_, Inicial_nAudios_,
                             Final_nAudios_, ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_,
                             Espectograma_, MFCC_)
            rutax2 = dar_ruta(ruta_resultados, 'x2_', Inicial_pNXML_, Final_pNXML_, Inicial_nAudios_,
                              Final_nAudios_, ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_,
                              Espectograma_, MFCC_)

        modelo=cargarModelo(rutaModelo, rutaPesos)
        matriz_confusion(modelo, titulo, rutax2, rutay)

    elif opcion==8:
        print('Menu para cambiar variables')
        print('-------------------------------------------')
        print('1. ventana_Tiempo_')
        print('2. Inicial_pNXML_')
        print('3. Final_pNXML_')
        print('4. nombre_')
        print('5. Unir x2')
        print('6. Espectogram_ o MFCC')
        print('7. validacion')
        print('-------------------------------------------')
        print('Porfavor seleccione una opción')

        entrada = int(input())

        while not (0 < entrada < 7):
            print('Porfavor seleccione una opción entre 1 y 8')
            entrada = int(input())

        if entrada == 1:
            pass
        elif entrada == 2:
            pass
    elif opcion==9:
        break
    else:
        break

print("Se acabo")