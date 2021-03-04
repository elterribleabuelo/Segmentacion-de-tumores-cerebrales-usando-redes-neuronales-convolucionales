import numpy as np

def metric_class(pred, label):
    """
    Calcula metricas (accuracy,especificidad y sensibilidad) para un ejemplo 
    en particular del WT(tumor completo)

    Parametros:
        pred (np.array): Array binario de predicciones(num classes, height, width, depth).
        label (np.array): Array binario de etqieutas(num classes, height, width, depth).

    Returns:
        accuracy(float): accuracy
        sensitivity (float): precision 
        specificity (float): recall 
    """

    # verdadero positivo
    tp = np.sum((pred == 1) & (label == 1))

    # verdadero negativo
    tn = np.sum((pred == 0) & (label == 0))
    
    # falso positivo
    fp = np.sum((pred == 1) & (label == 0))
    
    # falso negativo
    fn = np.sum((pred == 0) & (label == 1))

    # accuracy,sensibilidad,especificidad,coeficiente de dice
    accuracy = (tp+tn)/ (tp + tn + fp + fn)
    sensitivity = (tp)/ (tp + fn)
    specificity = (tn)/ (tn + fp)
    dice = (2*tp)/ ((2*tp)+fp+fn)

    return accuracy,sensitivity, specificity,dice