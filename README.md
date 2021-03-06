# Proyecto Gestos
## Descripción
El proyecto realizado trata sobre el reconocimiento de gestos de la mano con los cuales se logre la diferenciación de las acciones de soltar o arrojar a través de la librería de python mediapipe, la cual nos permite conocer los puntos exactos de la mano a través del trazo de líneas; también se hace uso de OpenCV la cual al igual que mediapipe es una librería de Python que nos permite el análisis de vídeos o imágnes para el reconocimiento de objetos a través de la visión artificial, recopilando y haciendo análisis de datos a través de otra librería llamada Numpy, para hacer uso de Math para realizar cálculos tales como "acos" que se refiere al arcocoseno con el cuál podemos hallar cálculos tales como el ángulo y el uso de degrees para la  transformación del ángulo de radianes a decimales.

## Procedimiento
Para llevar a cabo la creación del código debemos empezar con la creaciónn de los vídeos o imagenes a trabajar para la definición de los gestos con los cuales lograremos diferenciar las acciones, para ello se grabaron vídeos realizando dichas acciones, soltando un objeto, arrojando hacia abajo y arrojando hacia enfrente; con los vídeos ya obtenidos procedemoos a realizar el análisis de los posibles gestos que podemos obtener, reconociendo un grupo de 5 gestos entre las 3 acciones realizadas.

A continuación comenzamos con la creación del código donde en primer lugar debemos importar las librerías ya mencionadas y que serán usadas para el reconocimiento de este grupo de gestos.

En primera instancia debemos definir una función la cual contendrá el promedio de los puntos a tratar de la mano, tales como el promedio de los puntos de la muñeca, de los nudillos y de la punta de los dedos que serán usados más adelante para conocer la distancia existente entre ellos, es decir, entre la muñeca y la punta de los nudillos y la distancia entre la muñeca y la punta de los dedos, definiendo a través de operaciones que hacen uso de los promedios y distancias ya calculadas 2 de los gestos encontrados, mano abierta y mano cerrada.

A continuación debemos capturar las imagenes a tratar, en nuestro caso los archivos de vídeo grabados y obtener las dimensiones de estos, posteriormente se crea una variable para controlar los fps de las imagenes a tratar y que almacenarán los gestos, esto para poder controlar la velocidad del video, seguimos con la definición de las posiciones de nuestra mano en x, y, donde a través de estos definiremos 3 puntos y líneas que serán utiles para hallar ángulo de la mano durante la acción de arrojar, esto para la definición de un gesto.

![Angulo](Angulo.jpeg "Angulo")

Para hallar este ángulo hacemos uso de la librería math ya mencionada, donde hacemos uso de las funciones arcocoseno y "degrees" para hallar el valor del ángulo en un valor decimal, mientras hacemos uso de cv2 para el dibujo de las líneas que representarán nuestro ángulo y para la creación de los puntos de nuestro ángulo, también creando el contorno del mismo; siguiendo con nuestro proceso dibujamos nuestra mano con las líneas y ángulos definidos, creando una variable que almacenará los puntos de la mano.

Luego de obtener todos los puntos, líneas, posiciones, angulos necesitados hacemos uso de estos para la definición de nuestro conjunto de gestos, que se determinaron gracias al análisis de si al momento de realizarse la mano se encuentra cerrada o abierta y dependiendo el valor de su ángulo conocer su posición y por ende la acción realizada, donde finalmente al reconocer el gesto realizado escribe el nombre de este en pantalla.

## Gestos
### Mano Cerrada

![Mano_cerrada](Mano_cerrada.PNG "Gesto: Mano Cerrada")

### Mano Abierta - Soltar

![Mano_abierta](Mano_abierta.PNG "Gesto: Mano Abierta")

### Impulso

![Impulso](Impulso.PNG "Gesto: Impulso")

### Arrojado

![Arrojar](Arrojado.PNG "Gesto: Arrojado")

### Arrojado Enfrente

![Enfrente](Enfrente.PNG "Gesto: Arrojado Enfrente")
