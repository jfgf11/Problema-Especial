from metodos import *

while True:
    print('Menu para seleccionar la accion a ejecutar')
    print('-------------------------------------------')
    print('1. Descargar Audios y xml de google drive.')
    print('2. Sacar x, x2 y y')
    print('3. Descargar modelo de Drive')
    print('4. guardar modelo')
    print('5. Unir y')
    print('6. Unir x2')
    print('7. Crear RED 2D')
    print('8. Entrenar RED 2D')
    print('9. Salir')
    print('-------------------------------------------')
    print('Porfavor seleccione una opción')
    print('CUIDADO: algunas de estas operaciones consumen bastante '
        'memoria y pueden trabar su computador, antes de correr algo '
        'este seguro de que cuenta con la memoria suficiente !!!')

    opcion=int(input())

    while not (opcion>0 and opcion<10):
        print('Porfavor seleccione una opción entre 1 y 8')
        opcion=int(input())

    print('Usted selecciono la opción: ',opcion)

    ''''
    Parametros 
    '''

    ventana_Tiempo_ = 0.100  # La ventana de tiempo de cada muestra (XX_ms)
    salto_de_ventana_ = 4  # Corrimiento en tiempo (XXms/4)
    sample_rate_ = 11025  # Tasa remuestreo
    # Si se está obteniendo el espectogramo, el valor de la ventana no puede ser menor a 0.025s
    # Si se está obteniendo el MFCC, el valor de la ventana no puede ser menor a 0.03125s
    window_length_stft_ = 0.025  # Ventana de tiempo para la short-Time Fourier Transform
    Step_size_stft_ = 0.010  # Saltos el en tiempo para la transformada de Fourier, fíjenlo menor a la ventana stft, si quieren pueden aumentar

    Sin_Background_ = False  # True: no se obtienen datos de background; False: No se obtienen datos de background  NO MOVER
    Features_ = False  # Obtener o no features NO MOVER
    Inicial_pNXML_ = 1  # Número inicial de archivos XML utilizados  NO MOVER, a menos de que se quiera obtener 45 a 55
    Final_pNXML_ = 10  # Número final de archivos XML utilizados  NO MOVER, a menos de que se quiera obtener 45 a 55
    Inicial_nAudios_ = 0  # Número inicial de audios que se obtendrán (esto no aplica para nuestros audios)  NO MOVER
    Final_nAudios_ = 7  # Número final de audios que se obtendrán (esto no aplica para nuestros audios)  NO MOVER
    rutaDatosXML_ = "./drive/xml/"  # Ruta para encontrar archivos xml  NO MOVER
    rutaDatosSounds_ = "./drive/sounds/"  # Ruta para encontrar Audios  NO MOVER
    ruta_resultados = './drive/Datos_Procesados/datos'
    rutaDatosParciales_ = "./drive/parciales/datos"
    Solo_Background_ = False  # Solo obtener datos de background  NO MOVER
    nombre_ = "Casti"  ## PONER NOMBRE QUIEN REALIZA LA PRUEBA
    # a,b,c,d,e=str(ventana_Tiempo_)
    guardarLocal_ = False

    # Si ambos son True, se obtendrá únicamente el espectogram
    Espectogram_ = True  # Calcular el espectograma
    MFCC_ = False  # Calcular el MFCC
    numero = '11025_100ms_Spectogram'

    if opcion==1:
        guardarLocal_ = True
        hashGoogle() # Realiza conexion con Google Drive
        z = ObtenerSonidos(nombre=nombre_, Inicial_pNXML=Inicial_pNXML_, Final_pNXML=Final_pNXML_,
            Inicial_nAudios=Inicial_nAudios_, Final_nAudios=Final_nAudios_, ventana_Tiempo=ventana_Tiempo_,
            salto_de_ventana=salto_de_ventana_, Calcular_Features=Features_, Sin_Background=Sin_Background_,
            rutaDatosXML=rutaDatosXML_, rutaDatosParciales=rutaDatosParciales_,
            rutaDatosSounds=rutaDatosSounds_,
            Solo_Background=Solo_Background_, sample_rate=sample_rate_,
            window_length_stft=window_length_stft_,
            Step_size_stft=Step_size_stft_, guardarLocal=guardarLocal_, MFCC=MFCC_, Espectogram=Espectogram_)

    elif opcion==2:
        z = ObtenerSonidos(nombre=nombre_, Inicial_pNXML=Inicial_pNXML_, Final_pNXML=Final_pNXML_,
            Inicial_nAudios=Inicial_nAudios_, Final_nAudios=Final_nAudios_, ventana_Tiempo=ventana_Tiempo_,
            salto_de_ventana=salto_de_ventana_, Calcular_Features=Features_, Sin_Background=Sin_Background_,
            rutaDatosXML=rutaDatosXML_, rutaDatosParciales=rutaDatosParciales_,
            rutaDatosSounds=rutaDatosSounds_,
            Solo_Background=Solo_Background_, sample_rate=sample_rate_,
            window_length_stft=window_length_stft_,
            Step_size_stft=Step_size_stft_, guardarLocal=guardarLocal_, MFCC=MFCC_, Espectogram=Espectogram_)

    elif opcion==3:
        d = {}
        hashGoogle()  # Realiza conexion con Google Drive

        nombreModelo='Modelo_Casti_100.json'
        ruta = './drive/modelos/'
        guardarLocalmente(nombreModelo,ruta)
        nombreModelo = 'Pesos_Modelo_Casti_100.h5'
        ruta = './drive/modelos/'
        guardarLocalmente(nombreModelo, ruta)

    elif opcion==4:
        rutaModelo = './drive/modelos/Modelo_Casti_' + numero + '.json'
        rutaPesos = './drive/modelos/pesos/Pesos_Modelo_Casti_' + numero + '.h5'
        rutaDiagrama = './drive/modelos/diagrama/Diagrama_Modelo_Casti_' + numero + '.png'
        modelo=cargarModelo(rutaModelo, rutaPesos)
        guardarModelo(modelo, rutaModelo, rutaPesos, rutaDiagrama)

    elif opcion==5:
        print('Ingrese el numero de archivos que va a unir:')
        z=int(input())
        join('y', z, rutaDatosParciales_, ruta_resultados, Features_, Inicial_pNXML_, Final_pNXML_, Inicial_nAudios_,
            Final_nAudios_,
            ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_, MFCC_, Espectogram_)
    elif opcion==6:
        print('Ingrese el numero de archivos que va a unir:')
        z = int(input())
        join('x2', z, rutaDatosParciales_, ruta_resultados, Features_, Inicial_pNXML_, Final_pNXML_, Inicial_nAudios_,
            Final_nAudios_,
            ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_, MFCC_, Espectogram_)
    elif opcion==7:
        crear_red_2d(ruta_resultados, Features_, Inicial_pNXML_, Final_pNXML_, Inicial_nAudios_, Final_nAudios_,
                     ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_, MFCC_, Espectogram_,numero)
    elif opcion==8:
        rutaModelo = './drive/modelos/Modelo_Casti_' + numero + '.json'
        rutaPesos = './drive/modelos/pesos/Pesos_Modelo_Casti_' + numero + '.h5'
        modelo=cargarModelo(rutaModelo, rutaPesos)
        entrenarRed(modelo,ruta_resultados, Features_, Inicial_pNXML_, Final_pNXML_, Inicial_nAudios_, Final_nAudios_,
                     ventana_Tiempo_, sample_rate_, nombre_, Sin_Background_, Solo_Background_, MFCC_, Espectogram_)
    elif opcion==9:
        break
    else:
        print('wtf esto es ilegal.')

print("Se acabo")