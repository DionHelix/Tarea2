# Proyecto 2 - Red Neuronal Artificial - Reconocimiento Facial 

# Estrategia General y Descripción de las redes

Este proyecto tiene como fin crear una red neuronal artificial que sea
capaz de reconocer el rostro de una persona.

Primero se entrena la red con miles de imágenes de personas, donde se 
le especifican las características propias de cada persona, es decir, 
si se encuentra sonriendo, si tiene grandes labios, si tiene grandes
ojos, etc. Posteriormente con las características de esa red entrenada
con miles de imágenes, se guarda el modelo (donde se encuentran los pesos
y los "bias" propios del modelo) y se usa para entrenar otra red, que es
la que va a reconocer a la persona que le indiquemos. En este caso se
eligió utilizar el rostro de David Bowie para entrenar la segunda red.

Para realizar el primer entrenamiento se utilizó la base de datos CELEBA, 
la cual contiene cientos de miles de imágenes de celebridades, donde cada
imagen ha sido clasificada y se especifican con detalle las características
faciales individuales de cada persona mostrada. Para la segunda parte, se
utilizaron imágenes obtenidas del sitio oficial de facebook de David Bowie, 
y para comparar las imágenes se reutilizaron las imágenes de CELEBA.

Para este proyecto la guía primaria fueron las notas del curso, 
la documentación de keras, el foro stackoverflow.com y la documentación
que se puede hallar en tensorflow.org. 

Como estrategia general se siguieron los pasos mostrados en clase para
poder empezar la red y posteriormente con ayuda de la documentación y
los foros se fueron resolviendo los problemas que se presentaron. 

Cabe mencionar que el profesor encargado del curso fue de gran ayuda 
en la identificación de los problemas y en las propuestas de solución. 

El modelo usado en la primer red consta de varios sets de capas Conv2D, 
MaxPooling2D(2, 2), y Dropout(0.2), para una capa Flaten, dos densas, 
una de 100 neuronas y otra de 40, siendo que son 40 atributos por foto, 
y una capa con función de activación sigmoide. Esta red tiene un total de 
21 capas. 

Para la segunda red, se cargaron 18 capas del modelo de la red anterior, 
donde solo se reutilizaron las capas con estructura Conv2D, MaxPooling2D(2, 2), 
y Dropout(0.2), para luego incorporar nuevas capas densas, una de 100 y otra 
de una sola neurona, siendo que para esta segunda parte estamos comparando, 
así como una capa con función de activación sigmoide. Esta red tiene un total
de 21 capas.

# Problemas y Soluciones

Antes de comenzar el proceso de entrenamiento se decidió introducir 10 épocas, 
un total de 50 imágenes para entrenar, 50 de prueba y un batch size de 50,
esto para probar el modelo antes de comenzar con el entrenamiento real.

El primer problema que se tuvo fue cargar las etiquetas de cada imagen.
Este problema se presentó porque se quería utilizar un archivo con 
extensión CSV en lugar de la extensión TXT utilizada en clase. En ese
momento se creía que iba a ser más fácil trabajar con un archivo de
extensión CSV, pero dado que el tiempo se estaba acortando y no se lograba
avanzar en lo más básico del proyecto se opto por recurrir a un archivo de
extensión TXT.

El archivo TXT que proporciona CELEBA resultó tener dos espacios en la 
separación de la información, por lo que se preparó un archivo que eliminó
ese espacio, ese archivo se utilizó para etiquetar las imágenes. La información
contenida en el archivo preparado contiene los valores 1 y -1.

Debido al uso del archivo con extensión CSV, no incluimos parte del código 
mostrado en clase que era crucial para la separación de las etiquetas. Se 
encontraron errores de tipo Tensor (ver "ValueError, TypeError. png"), así
como errores de sintaxis (ver "Key Error.png").

Al principio se creía que la versión de Python era la responsable, pero 
simplemente se debía la extensión del archivo con el que se estaba trabajando
y que no se especificó que queríamos separar la data por comas. (ver "Error 
Convertir de Numpy a Tensor.pdf" y "Type Error.png").

Una vez hecho ese cambio se encontró con otro error (ver "Modelo no Compila.png").
Se tenía un error de dimensiones, pero no se entendió esto hasta que se modificó 
la red, se quitó capa tras capa del modelo y se veía que el error permanecía
(ver "ValueError - Input 0 of layer sequential9 is incompatible with the layer.png" y
"ValueError - Input 0 of layer maxpooling2d4 is incompatible with the layer.png").

Se encontró que KERAS, al especificar un "input_shape" no incorporaba la entrada 
correspondiente al batch size, por lo que detectaba menos dimensiones de las que
debía. Para Solucionar esto se especificó un batch size directamente en el dataset 
"labeled_images=data.map(process_file)" cambiándolo como "labeled_images 
= data.map(process_file).batch(10)". Con esto se había superado ese error, pero 
aun no compilaba el modelo, fue hasta que se incorporó un paréntesis en la
función de costo, de modo que ahora se podía indicar un argumento a la misma.

Con estos cambios la red ya lograba compilar, pero no aprendía. (ver "Ya 
logra entrenar - Demasiado tiempo por época.png"). Se comenzó a usar una 
función de Costo Sparce Categorical Crossentropy y luego pasamos a una función
Categorical Crossentropy agregándole reducción en el argumento. Pensando que 
tal vez la red se estuviese saturando se quitaron algunas capas MaxPooling2D, 
no funcionó. Se intentó cambiar el optimizador, en busca de alguna novedad, 
pero nada funcionaba. Incluso se obtenía una función de costo con valores
negativos. Se creyó que cambiando el batch size a 128 se vería algún cambio
o alguna pista que pudiera darnos idea de donde hay un error.

Fue hasta que se encontró que no se habían cambiado los valores de la data 
preparada a ceros y unos, una vez realizado ese cambio la función de costo
bajó dramáticamente. (ver "Ya logra entrenar - No Aprende - Loss Function 
baja.png"). Cambiando la función de costo a Mean Squared Error ya se alcanzaba
una precisión del 80%, esto mejoró aun más al cambiar a Binary Crossentropy y 
las "metrics" a Binary Accuracy, sin embargo, debido a la poca data que se 
había usado los pasos caían fuera de la misma, por lo que se obtuvo un error.
(ver "Ya logra entrenar - Ya aprende - Loss Function baja, error de Batchsize.png")

Para solucionar esto se implementaron dentro de la compilación del modelo los 
parámetros "steps per epoch" y "validation_steps" con la siguiente estructura
(ver "Steps-Validation per Epoch.png"). Con eso se aseguró que el error del
batchsize nunca pudiese ocurrir otra vez. Con un batchsize de 128 solo se
obtenían 6 pasos por época, así que se regresó a 50. Se aumentaron las imágenes,
ahora usando 18000 para entrenar y 2000 de prueba, se aumentaron las épocas a 25. 
Con estos cambios se obtenían 81 pasos por época, se alcanzó una precisión del 86% 
y un valor de la función de costo de 0.3036. (ver "Ya logra entrenar - Ya aprende 
- Modelo Puede ser mejorado.png")

Este modelo con 25 épocas se guardó como "FaceRecog.h5".

Se observó una tendencia de crecimiento en la precisión y una disminución de la
función de costo conforme aumentaban las épocas, así que pusimos a entrenar el 
modelo con 50 épocas. Aunque si se observó una mejoría no fué tan grande. 
(ver "Modelo Parte 1 50 Épocas.png").

Este modelo con 50 épocas se guardó como "FaceRecogPlus.h5".

En la segunda red nuevamente se hicieron pruebas con 10 épocas, 70 imágenes para
entrenar, 34 de prueba, batch size de 50 antes del entrenamiento real.

Para la segunda red se encontró un error en el entrenamiento (ver "Segunda Parte 
- Error en la función Train.png"). Esto se debió a la longitud de la data de
entrenamiento, era demasiado corta, por lo que el parámetro "steps per epoch"
impedía que la red entrenara. Este error se solucionó quitando el parámetro. 
Se encontró que una precisión de 88% y un valor de la función de costo de 0.2927.
(ver "Entrenamiento parte 2.png", "Loss vs Val-Loss.png" y "Precisión vs
Val-Precisión.png").

Se aumentaron las épocas a 50 buscando una mejoría en los resultados. Se encontró 
una precisión de 88% y un valor de la función de costo de 0.5388. Cabe mencionar que
estos valores oscilaron mucho durante el entrenamiento. (ver "Loss vs Val-Loss 50 
épocas.png ", "Precisión vs Val-Precisión 50 épocas.png" y "Modelo Parte 2 50 Épocas.png").

Todo estaba funcionando tan bien que incluso parecía sospechoso. 

Se evaluó el modelo para esta segunda parte, donde se muestra que la función de costo
alcanzo un 0.4914 y la precisión un 88%, también se imprimió en terminal el valor de
predicción de un set de imágenes de prueba. (ver "Modelo Evaluado y Modelo con Predicción .png")

En lo que sigue se incorporaron líneas de código que permitían imprimir imágenes 
de este set de prueba donde se indica si son David Bowie o alguien más.
(ver la carpeta "Capturas Test Model Predict"). Ahí se encontró que la red identificaba
todas las imágenes como David Bowie.

Al principio se creyó que el error de código, donde se omitió poner en las etiquetas
otro valor que no fuera David Bowie. Pero se encontró que la red identificaba
todas las imágenes de ese set de prueba como David Bowie, se ve que la mayoría son 
su persona, pero hay un margen considerable de imágenes que no corresponden, ahí es
donde se ve que el valor de la función de costo influye mucho, además de
que la precisión no llegó al 90%. El modelo puede ser mejorable, lo que se debe buscar
es reducir la función de costo aún más, tanto del modelo cargado como del nuevo modelo.
(ver la carpeta "Capturas Finales Test Model Predict").
.
# Conclusiones

Los errores encontrados se deben más a la poca experiencia con el código y al
desconocimiento del mismo. Sin embargo todos los errores que arrojó la terminal
fueron resueltos con éxito. 

Los resultados fueron relativamente satisfactorios, ya que la red puede ser
mejorada aún más. Es primordial reducir bajo todos los medios el valor de la
función de costo en la primer parte del proyecto, ya que influye demasiado en
la segunda parte del mismo. 

Para reducir el tiempo de entrenamiento de la primer parte del proyecto se pueden
preprocesar las imágenes con otros métodos que hagan que los parámetros de
entrenamiento se reduzcan aún más.

El reto de construir esta red fue considerable, incluso hubo momentos donde se
creyó que el proyecto no iba a ser concluido.

Se cree que si se implementa la librería CUDA se obtendrán resultados más rápido.