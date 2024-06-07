import torch
from torch.utils.data import Dataset
import numpy as np
import os


class SynthDataLoader(Dataset):
    def __init__(
        self,
        data_path,
        mode,
        params={
            "same_data_norm": False,
            "no_data_norm": False,
        },
    ):
        self.mode = mode
        self.data_path = data_path
        if self.mode == "train":
            path = os.path.join(data_path, "train_feats")
            edge_path = os.path.join(data_path, "train_edges")
        elif self.mode == "val":
            path = os.path.join(data_path, "val_feats")
            edge_path = os.path.join(data_path, "val_edges")
        elif self.mode == "test":
            path = os.path.join(data_path, "test_feats")
            edge_path = os.path.join(data_path, "test_edges")
        self.feats = torch.load(path)
        self.edges = torch.load(edge_path)
        self.same_norm = params["same_data_norm"]
        self.no_norm = params["no_data_norm"]
        if not self.no_norm:
            self._normalize_data()

    def _normalize_data(self):
        train_data = torch.load(os.path.join(self.data_path, "train_feats"))
        if self.same_norm:
            self.feat_max = train_data.max()
            self.feat_min = train_data.min()
            self.feats = (self.feats - self.feat_min) * 2 / (
                self.feat_max - self.feat_min
            ) - 1
        else:
            self.loc_max = train_data[:, :, :, :2].max()
            self.loc_min = train_data[:, :, :, :2].min()
            self.vel_max = train_data[:, :, :, 2:].max()
            self.vel_min = train_data[:, :, :, 2:].min()
            self.feats[:, :, :, :2] = (self.feats[:, :, :, :2] - self.loc_min) * 2 / (
                self.loc_max - self.loc_min
            ) - 1
            self.feats[:, :, :, 2:] = (self.feats[:, :, :, 2:] - self.vel_min) * 2 / (
                self.vel_max - self.vel_min
            ) - 1

    def unnormalize(self, data):
        if self.no_norm:
            return data
        elif self.same_norm:
            return (data + 1) * (self.feat_max - self.feat_min) / 2.0 + self.feat_min
        else:
            result1 = (data[:, :, :, :2] + 1) * (
                self.loc_max - self.loc_min
            ) / 2.0 + self.loc_min
            result2 = (data[:, :, :, 2:] + 1) * (
                self.vel_max - self.vel_min
            ) / 2.0 + self.vel_min
            return np.concatenate([result1, result2], axis=-1)

    def __getitem__(self, idx):
        return {"inputs": self.feats[idx], "edges": self.edges[idx]}

    def __len__(self):
        return len(self.feats)
    

"""La clase SynthDataLoader hereda de Dataset, permitiendo su uso con cargadores de datos (DataLoader) en PyTorch.

Método Constructor __init__

El constructor inicializa la clase con las siguientes operaciones:

Asignación de Parámetros: Almacena el modo (train, val o test) y la ruta de los datos.
Carga de Datos: Carga las características (feats) y los bordes (edges) desde los archivos correspondientes según el modo.
Normalización de Datos: Llama a _normalize_data para normalizar los datos si no_data_norm es False.

Método _normalize_data

Este método normaliza los datos de entrenamiento:

Si same_data_norm es True, normaliza todos los datos utilizando el valor máximo y mínimo de todo el conjunto de datos.
Si same_data_norm es False, normaliza las posiciones (loc) y velocidades (vel) por separado utilizando sus propios valores máximos y mínimos.

Método unnormalize

Este método desnormaliza los datos:

Si no_norm es True, devuelve los datos sin cambios.
Si same_norm es True, desnormaliza los datos utilizando los valores máximos y mínimos de todo el conjunto de datos.
Si same_norm es False, desnormaliza las posiciones y velocidades por separado.

Método __getitem__

Este método devuelve un diccionario con las características y bordes correspondientes al índice idx.

Método __len__

Este método devuelve la longitud del conjunto de datos.

Esta clase es esencial para cargar y procesar los datos sintéticos utilizados en los experimentos del artículo, asegurando 
que los datos estén en el formato adecuado para ser utilizados en los modelos de aprendizaje automático."""




class LoaderWrapper:
    def __init__(self, dataloader, n_step):
        self.step = n_step
        self.idx = 0

        self.loader = dataloader
        self.iter_loader = iter(self.loader)

    def __iter__(self):
        return self

    def __len__(self):
        return self.step

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)
        
""" Definición de la Clase LoaderWrapper

El constructor inicializa la clase con las siguientes operaciones:

Asignación de Parámetros:
dataloader: El dataloader original que se desea envolver.
n_step: El número de pasos deseados para iterar.
Inicialización de Contadores:
self.step: Número de pasos a iterar.
self.idx: Contador para rastrear el número de pasos actuales.
Inicialización del Iterador:
self.loader: Almacena el dataloader.
self.iter_loader: Crea un iterador a partir del dataloader.

Método __iter__

Este método permite que la clase LoaderWrapper sea un iterador, retornando self.

Método __len__

Este método devuelve el número de pasos (n_step) que el LoaderWrapper está configurado para iterar.

Método __next__

Este método permite obtener el siguiente elemento del iterador. La lógica es la siguiente:

Comprobación de Pasos:
Si el contador self.idx ha alcanzado self.step, reinicia el contador (self.idx = 0) y lanza una excepción StopIteration para detener la iteración.
Si no, incrementa el contador (self.idx += 1).
Obtención del Siguiente Elemento:
Intenta obtener el siguiente elemento del iterador self.iter_loader.
Si se alcanza el final del iterador (StopIteration), reinicia el iterador (self.iter_loader = iter(self.loader)) y obtiene el siguiente elemento del nuevo iterador.

Uso de LoaderWrapper
Esta clase es útil para controlar el número de pasos que se desea iterar sobre un dataloader en PyTorch. En lugar de iterar indefinidamente o hasta el final
del dataloader, LoaderWrapper permite especificar un número fijo de pasos (n_step), reiniciando el iterador cuando sea necesario para cumplir con el número de pasos deseado. Este comportamiento es especialmente útil en escenarios donde se 
necesita un número constante de iteraciones, como en ciertas configuraciones de entrenamiento o evaluación en aprendizaje automático."""
