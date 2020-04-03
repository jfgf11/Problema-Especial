# Problema-Especial

## Proyecto
Repositorio de clase problema especial, en el que se 
implementa una modelo de aprendizaje para detectar sonidos
como gritos, disparos, vidrios rotos y sonido de ambiente. 

## Instalación dependencias

A partir del archivo requerements.txt (aun no funciona) 
podra descargar 
las dependencias necesarias para ejecutar el proyecto, 
este archivo lo puede correr desde la raiz del proyecto
con el siguiente comando: 

`````
# If pip is not already installed run:
sudo apt install python3-pip

# Install requirements globally
sudo pip3 install -r ./requirements.txt
`````
De manera alterna tambien puede instalar las librerias 
necesarias por su cuenta para esto siga las instrucciones
quw se muestran en seduida:

1.Asegurese de tener instalado python y cree un ambiente
 virtual desde el cual se va a ejecutar el proyecto:
````
sudo apt update
sudo apt upgrade
sudo apt install python3.7
````
Para asegurarse de que la instalación se realizara de 
manera correcta ejecute:

````
python3 --version
````

A continuación se debe crear el ambiente virtual en el que
se instalaran las librerias necesarias para la ejecución 
del proyecto

````
 python3 -m pip install --upgrade pip
sudo pip3 install virtualenv
virtualenv -p python3 venv_problema
````
Para acceder al ambiente virtual:
````
source venv_problema/bin/activate
````
Dentro del ambiente ejecute los siguientes comandos:
````
pip install numpy
pip install pyaudio
pip install librosa
pip install --upgrade tensorflow
````