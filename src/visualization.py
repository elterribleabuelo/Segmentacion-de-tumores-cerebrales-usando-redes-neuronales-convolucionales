import cv2
import h5py
import imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical



def get_labeled_image(image, label, is_categorical=False):
    """
    Función que permite tener en la misma IMR las respectivas máscaras binarizadas
    Donde los píxeles blancos dentro de la zona en negro(tumor completo), serán las
    partes que conforman el tumor(Nuleo,edema,tumor realzado)
    Parameters
    ----------
    
    image : volumen de entrada con las 4 secuencias
    label : máscara de segmentación
    is_categorical : False

    Returns
    -------
    labeled_image : secuencia FLAIR de la IMR donde los píxeles 0 forman el tumor completo,
    y además pixeles = 255 que se encuentran dentro del tumor completo
    son las partes del tumor (Nucleo,edema,tumor completo)
    Tiene como dimensiones : (240,240,155,3)
    La última dimensión está formada por los canales RGB.
    """
    
    if not is_categorical:
        # num_classes: WT,ED,ET,TC
        # label debe estar formado por 0 y 1
        label = to_categorical(label, num_classes=4).astype(np.uint8)
    
    # Escogemos la secuencia FLAIR de la IMR
    image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    
    # Inicializamos el contenedor
    labeled_image = np.zeros_like(label[:, :, :, 1:])

    # Multiplicamos la secuencia FLAIR por el tumor completo
    # labeled_image estará formado por pixeles de 0-255
    # donde los pixeles= 0 serán los de la clase tumor
    
    labeled_image[:, :, :, 0] = image * (label[:, :, :, 0]) # Núcleo del tumor (R)
    labeled_image[:, :, :, 1] = image * (label[:, :, :, 0]) # Edema peritumoral (G)
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0]) # Tumor realzado (B)
    
    # Dentro del tumor completo , sobreescribimos cada una de sus partes contenidas
    # en el array "label"
    
    labeled_image = labeled_image + label[:, :, :, 1:] * 255 
    
    return labeled_image
    

def visualize_data_gif(data):
    """
    Visualizacion de máscaras de segmentación en los diversos cortes
    de una resonancia magnética
    Parameters
    ----------
    data_ : Arreglo formado por píxeles de 0-255 , donde los píexeles 0 forman el tumor
    completo y los píxeles 1 dentro del bloque del tumor completo son las partes del tumor
    # Verde: edema peritumoral
    # Azul: tumor realzado
    # Rojo: nucleo del tumor

    Returns
    -------
    TYPE
        imagen en formato de gif
    """
    images = []
    for i in range(data.shape[2]):
        x = data_[min(i, data.shape[0] - 1), :, :] # (240,155,3) # Plano YZ
        y = data_[:, min(i, data.shape[1] - 1), :] # (240,155,3) # Plano XZ
        z = real[:, :, min(i, data.shape[2] - 1)] # (240,240,3) # Plano XY
        img = np.concatenate((x,y,z), axis=1)
        images.append(img)
    imageio.mimsave("/tmp/gif.gif", images, duration=0.01)
    return Image(filename="/tmp/gif.gif", format='png')



def visualize_patch(X, y):
    """
    Visualizacion de subvolumenes

    Parameters
    ----------
    X : Subvolumen de entrada
    y : máscaras de predicción

    Returns
    -------

    """
    fig, ax = plt.subplots(1, 2, figsize=[10, 5], squeeze=False)

    ax[0][0].imshow(X[:, :, 0], cmap='Greys_r')
    ax[0][0].set_yticks([])
    ax[0][0].set_xticks([])
    ax[0][1].imshow(y[:, :, 0], cmap='Greys_r')
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])

    fig.subplots_adjust(wspace=0, hspace=0)
    

def predict_and_viz(image, label, model, threshold = 0.5):
    """
    Función encargada de realizar la segmentación total de la resonancia
    magnética :(240,240,155) , además genera un par de videos donde se compara
    la segmentación de un experto vs la hecha por la CNN

    Parameters
    ----------
    image : array con las 4 secuencias.
    label : array con las 3 máscaras
    model : red neuronal encargada de la predicción
    threshold: umbral para la clasificación
    Returns
    -------
    out: Video .mp4 en el que se observa los diversos cortes en el eje Z de la IMR
    para cada una de las máscaras (TC,ED,ET)

    out_wt: Video .mp4 en el que se observa los diversos cortes en el eje Z de la IMR
    para el tumor completo (WT)
    
    En ambos videos se muestra un versus de la segmentación realizada por la CNN y 
    la segmentación realizada por el experto.
    """
    image_labeled = get_labeled_image(image.copy(), label.copy())

    # Dimensiones del contenedor

    dim_1 = max([x+160 for x in range(0,image.shape[0],160)])
    dim_2 = max([y + 160 for y in range(0,image.shape[1],160)])
    dim_3 = max([z+16 for z in range(0,image.shape[2],16)])

    # Inicializamos el contenedor de la IMR completa

    model_label = np.zeros([3, dim_1, dim_2, dim_3]) #shape:(3,320,320,160)
    model_label_bin = np.zeros([3, dim_1, dim_2, dim_3]) #shape:(3,320,320,160)

    # Recorremos todos los subvolúmenes que se puedan generar
    # y los agregamos al contenedor principal
    for x in range(0, image.shape[0], 160):
        for y in range(0, image.shape[1], 160):
            for z in range(0, image.shape[2], 16):
                # Inicializamos el contenedor
                patch = np.zeros([4, 160, 160, 16])

                # Movemos el canal de las secuencias de la resonancia a la primera posicion
                p = np.moveaxis(image[x: x + 160, y: y + 160, z:z + 16], 3, 0)

                # Movemos el contenido de p a patch para que pueda ser procesado por la CNN
                patch[:, 0:p.shape[1], 0:p.shape[2], 0:p.shape[3]] = p

                # Predecimos sobre el bloque de tamaño :(3,160,160,16)
                pred = model.predict(np.expand_dims(patch, 0))

                # Inicializamos contenedor con las máscaras binarizadas
                pred_bin = pred
                pred_bin[pred_bin > threshold] = 1.0  # clase tumor
                pred_bin[pred_bin <= threshold] = 0.0 # clase no tumor

                # Agregamos la región predicha al contenedor principal
                model_label[:, x:x + p.shape[1], y:y + p.shape[2], z: z + p.shape[3]] += pred[0][:, :p.shape[1], :p.shape[2],:p.shape[3]]
                model_label_bin[:, x:x + p.shape[1], y:y + p.shape[2], z: z + p.shape[3]] += pred_bin[0][:, :p.shape[1], :p.shape[2],:p.shape[3]]


    # Damos la forma del label original : (3,240,240,155)
    model_label = np.moveaxis(model_label[:, 0:240, 0:240, 0:155], 0, 3)
    model_label_bin = np.moveaxis(model_label_bin[:, 0:240, 0:240, 0:155], 0, 3)
    # Nota: model_label almacena las máscaras(TC,ED,ET) en forma de RGB
    # y model_label_bin almacena las máscaras (TC,ED,ET) en formato de 0 y 1

    # Hallamos la predicción del tumor completo como la suma de las máscaras
    model_label_bin_WT = model_label_bin[:,:,:,0] + model_label_bin[:,:,:,1] + model_label_bin[:,:,:,2]

    # Máscara final predicha del tumor completo(WT)!!!!!!
    model_label_bin_WT = np.where(model_label_bin_WT >= 1, 1, 0)

    
    # Inicializamos el contenedor con 4 máscaras las cuales serán:
    # WT(verdadera) , TC,ED,ET (predichas por la CNN)
    model_label_reformatted = np.zeros((240, 240, 155, 4))
    model_label_bin_reformatted = np.zeros((240,240,155))

    # Guardamos la máscara del tumor completo
    model_label_reformatted = to_categorical(label, num_classes=4).astype(np.uint8)

    # Guardamos las máscaras predichas por la CNN(TC,ED,ET)
    model_label_reformatted[:, :, :, 1:4] = model_label

    # Guardamos la máscara verdadera del tumor completo(WT)
    model_label_bin_reformatted = model_label_reformatted[:,:,:,0]

    # Invertimos los valores de los pixeles ya que en TC están con valor de 0
    model_label_bin_reformatted = np.where(model_label_bin_reformatted==1,0,1) # Máscara final verdadera del WT!!!!!!


    # Secuencia FLAIR con las máscaras(TC,ED,ET)
    model_labeled_image = get_labeled_image(image, model_label_reformatted,
                                            is_categorical=True)
    
    # Secuencia FLAIR  con el tumor completo predicho(WT)!!!!!
    image_flair_wt = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    labeled_image_wt_pred = np.zeros_like(model_label_bin_WT)
    labeled_image_wt_pred = np.where(model_label_bin_WT==1,255,image_flair_wt)


    # Secuencia FLAIR  con el tumor completo verdadero(WT)!!!!
    labeled_image_wt_real = np.zeros_like(model_label_bin_reformatted)
    labeled_image_wt_real = np.where(model_label_bin_reformatted==1,255,image_flair_wt)

    #### Generamos videos y gif sobre el plano XY ####

    images = []
    images_bin =[]
    area_expert =[]
    area_cnn = []
    count = 0
    count_bin = 0
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('Brats2020.mp4',fourcc, 10.0, (680,240))
    out_wt = cv2.VideoWriter('Brats2020_WT.mp4',fourcc, 10.0, (680,240),0)
    font = cv2.FONT_ITALIC

    #### Para las máscaras TC,ED,ET ####
    for i in range(model_labeled_image.shape[2]):
        pred = model_labeled_image[:, :, min(i, model_labeled_image.shape[2] - 1)] # (240,240,3)
        real = image_labeled[:, :, min(i, model_labeled_image.shape[2] - 1)] # (240,240,3)
        aux =  np.zeros([240, 200, 3]).astype(np.uint8)
        img = np.concatenate((real,pred,aux), axis=1)
        images.append(img)
    imageio.mimsave("./Brats2020.gif", images, duration=0.01)

    #### Para la máscara WT ####
    for j in range(labeled_image_wt_real.shape[2]):
        pred_wt = labeled_image_wt_pred[:, :, min(j, labeled_image_wt_real.shape[2] - 1)].astype(np.uint8) # (240, 240)
        real_wt = labeled_image_wt_real[:, :, min(j, labeled_image_wt_real.shape[2] - 1)].astype(np.uint8) # (240, 240)
        aux_wt =  np.zeros([240, 200]).astype(np.uint8)
        img_wt = np.concatenate((real_wt,pred_wt,aux_wt), axis=1)
        images_bin.append(img_wt)
    imageio.mimsave("./Brats2020_WT.gif", images_bin, duration=0.01)
  
  
    #### Para las máscaras TC,ED,ET ####
    for frame in images:
      cv2.putText(frame,"Experto", (85,30), 0, 0.8, (255,255,255))
      cv2.putText(frame,"Unet", (320,30), 0, 0.8, (255,255,255))
      cv2.putText(frame,str("Capa:") + str(count),(500,30),0,0.5,(255,255,255))

      cv2.putText(frame,str("Nucleo del tumor:"),(500,80),0,0.25,(255,255,255))
      cv2.rectangle(frame,pt1 = (600,70),pt2 =(620,90),color = (255,0,0),thickness = -1)

      cv2.putText(frame,str("Edema peritumoral:"),(500,140),0,0.25,(255,255,255))
      cv2.rectangle(frame,pt1 = (600,130),pt2 =(620,150),color = (0,255,0),thickness = -1)

      cv2.putText(frame,str("Tumor realzado:"),(500,200),0,0.25,(255,255,255))
      cv2.rectangle(frame,pt1 = (600,190),pt2 =(620,210),color = (0,0,255),thickness = -1)

      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      out.write(frame)
      count = count + 1
    out.release()

    #### Para la máscara WT ####
    for frame_bin in images_bin:

      cv2.putText(frame_bin,"Experto", (85,30), 0, 0.8, (255,255,255))
      cv2.putText(frame_bin,"Unet", (320,30), 0, 0.8, (255,255,255))
      cv2.putText(frame_bin,str("Capa:") + str(count_bin),(500,30),0,0.5,(255,255,255))

      cv2.putText(frame_bin,str("Tumor completo:"),(500,80),0,0.25,(255,255,255))
      cv2.rectangle(frame_bin,pt1 = (600,70),pt2 =(620,90),color = (255,255,255),thickness = -1)

      out_wt.write(frame_bin)
      count_bin = count_bin + 1
    out_wt.release()

    print("Videos guardados correctamente")

    return out,out_wt