# Proyecto Gestos
## Descripción
El proyecto realizado trata sobre el reconocimiento de gestos de la mano con los cuales se logre la diferenciación de las acciones de soltar o arrojar a través de la librería de python mediapipe, la cual nos permite conocer los puntos exactos de la mano a través del trazo de líneas; también se hace uso de OpenCV la cual al igual que mediapipe es una librería de Python que nos permite el análisis de vídeos o imágnes para el reconocimiento de objetos a través de la visión artificial, recopilando y haciendo análisis de datos a través de otra librería llamada Numpy, para hacer uso de Math para realizar cálculos tales como "acos" que se refiere al arcocoseno con el cuál podemos hallar cálculos tales como el ángulo y el uso de degrees para la  transformación del ángulo de radianes a decimales.

## Procedimiento
Para llevar a cabo la creación del código debemos empezar con la creaciónn de los vídeos o imagenes a trabajar para la definición de los gestos con los cuales lograremos diferenciar las acciones, para ello se grabaron vídeos realizando dichas acciones, soltando un objeto, arrojando hacia abajo y arrojando hacia enfrente; con los vídeos ya obtenidos procedemoos a realizar el análisis de los posibles gestos que podemos obtener, reconociendo un grupo de 5 gestos entre las 3 acciones realizadas.

A continuación comenzamos con la creación del código donde en primer lugar debemos importar las librerías ya mencionadas y que serán usadas para el reconocimiento de este grupo de gestos.
