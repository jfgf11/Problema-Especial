#!/usr/bin/env python3
import unittest
from xml.dom import minidom


def test_find_input_device():
    return 0
    #return lap.find_input_device()

def test_process_block(audio):
    return [1]

def leer_audios(ruta_audios):
    doc = minidom.parse(ruta_audios)
    # Vector que contiene el tiempo en segundos de inicio de todos los eventos
    start = doc.getElementsByTagName("STARTSECOND")
    # Vector que contiene el tiempo en segundos de finalizacion de todos los eventos
    finish = doc.getElementsByTagName("ENDSECOND")
    # Vector que contiene la etiqueta de cada uno de los eventos
    ID = doc.getElementsByTagName("CLASS_ID")
    # Indica informacion de todos los eventos en un archivo xml (tamaño)
    events = doc.getElementsByTagName("events")
    # Se obtiene el numero de eventos en un audio
    a, b, c, d = (events[0].attributes["size"].value)
    # numero de eventos en un audio
    nEventos = int(c + d)
    
    audios=[]
    return audios
ruta_audios = './audios_test.xml'

class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(test_find_input_device(), 2)
        audios = leer_audios(ruta_audios)
        for i in audios:
            self.assertEqual(test_process_block(),[1])