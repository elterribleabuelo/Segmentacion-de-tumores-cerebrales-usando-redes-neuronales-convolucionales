# Segmentación de tumores cerebrales en imágenes de resonancia magnética usando redes neuronales convolucionales

El repositorio contiene arquitecturas basadas en redes neuronales convolucionales 3D usando el dataset <a href="https://www.kaggle.com/awsaf49/brats20-dataset-training-validation">BraTS2020</a>, presentado como proyecto final del curso de Inteligencia Artificial Avanzada.

<img src="https://github.com/renzoguerrero17/Segmentacion-de-tumores-cerebrales-usando-redes-neuronales-convolucionales/blob/master/assets/Brats2020.gif" width="60%" height="60%" style="max-width:100% alt="centered image">

*Pasos*:

- **1.Analisis exploratorio**: Se exploró que secuencias conforman una resonancia magnética, además de seleccionar que máscaras de segmentación se usarán en el conjunto de llegada las cuales seran el Edema Peritumoral(ED),Núcleo del tumor(TC) y Tumor realzado(TE).

- **2.Preprocesamiento**: Se procedió a juntar las secuencias FLAIR,T1,T2  Y T1 ponderada ya que en el dataset original se encuentran separadas.

   El dataset preprocesado se encuentra en las siguientes rutas:

   A) Resonancias mágneticas: https://drive.google.com/drive/folders/1_a0f019pz7BQ5GCWjgSEIZDcDyyf1AAf?usp=sharing

   B) Máscaras segmentadas: https://drive.google.com/drive/folders/1e-4ee-X8br3TXzT7Wsq9LjFo_yby4g69?usp=sharing

- **3.Generación de subvolúmenes**: Una vez unidas las secuencias procedemos a submuestrear el volumen total de la resonancia magnética en bloques de 160x160x16, además estos subvolúmenes serán guardados en formatos .h5 para su posterior uso, no sin antes normalizar las imágenes de resonancia magnética a un rango de [0-1].
Se tomo un promedio de 25 subvolúmenes por cada resonancia mágnetica perteneciente a un paciente.

  El dataset final en formato .h5 se encuentra en la siguiente ruta: https://drive.google.com/drive/folders/1_xMn9bkxl7NoUwE69j25xsglOsMS1sv3?usp=sharing
