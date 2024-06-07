import numpy as np


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


"""La función encode_onehot convierte una lista de etiquetas en una representación one-hot. La codificación one-hot es una forma común de representar datos categóricos como vectores binarios, donde cada posición en el vector corresponde a una clase y el valor es 1 para la clase presente y 0 para todas las demás.

Parámetros
labels: Una lista de etiquetas (clases) que se desea convertir a una representación one-hot.
Funcionamiento
classes = set(labels): Extrae las clases únicas de la lista de etiquetas y las almacena en un conjunto. Esto asegura que cada clase única se represente solo una vez.

classes_dict: Crea un diccionario donde cada clase única se asigna a una fila de una matriz de identidad de tamaño len(classes). La matriz de identidad tiene unos en la diagonal principal y ceros en todas las demás posiciones.

np.identity(len(classes))[i, :] genera una fila de la matriz de identidad correspondiente a la clase c en la posición i.
{c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)} mapea cada clase c a su correspondiente vector one-hot.
labels_onehot: Mapea cada etiqueta en labels a su vector one-hot correspondiente utilizando el diccionario classes_dict.

list(map(classes_dict.get, labels)) convierte cada etiqueta en su vector one-hot.
np.array(..., dtype=np.int32) convierte la lista de vectores one-hot en un array de NumPy con el tipo de dato int32.
return labels_onehot: Retorna el array de NumPy que contiene la representación one-hot de las etiquetas.

La función encode_onehot es útil en el preprocesamiento de datos categóricos para modelos de aprendizaje automático. La representación one-hot es esencial en muchas tareas de aprendizaje supervisado, ya que los modelos requieren entradas numéricas y no pueden trabajar directamente con datos categóricos. 
Esta función permite convertir etiquetas categóricas en una forma adecuada para su uso en redes neuronales y otros algoritmos de aprendizaje automático."""