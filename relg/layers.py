import numpy as np

import torch.nn as nn


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(
        self,
        n_in,
        n_hid,
        n_out,
        do_prob=0.0,
        no_bn=False,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True),
        )
        if no_bn:
            self.bn = None
        else:
            self.bn = nn.BatchNorm1d(n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        # Input shape: [num_sims, num_objects, num_features]
        x = self.model(inputs)
        if self.bn is not None:
            return self.batch_norm(x)
        else:
            return x
        

"""La clase MLP (Multi-Layer Perceptron) define una red neuronal completamente conectada con dos capas, funciones de activación ELU y normalización por lotes opcional. 
Es una clase general que puede ser utilizada en diversos modelos de aprendizaje profundo para transformar entradas de datos a través de varias capas lineales con activaciones no lineales y regularización por dropout.

Atributos
model: Una secuencia de capas lineales y funciones de activación ELU con dropout.
bn: Una capa de normalización por lotes opcional. Si no_bn es True, se desactiva la normalización por lotes.
Métodos
__init__: Inicializa la clase MLP con las dimensiones de entrada, las dimensiones ocultas y las dimensiones de salida, así como la probabilidad de dropout y una bandera para desactivar la normalización por lotes.

n_in: Dimensión de entrada.
n_hid: Dimensión oculta.
n_out: Dimensión de salida.
do_prob: Probabilidad de dropout.
no_bn: Bandera para desactivar la normalización por lotes.
Configura una secuencia de capas lineales con funciones de activación ELU y dropout.
Configura una capa de normalización por lotes opcional.
Llama a init_weights para inicializar los pesos de las capas.
init_weights: Inicializa los pesos de las capas lineales utilizando la inicialización de Xavier. También inicializa los parámetros de la capa de normalización por lotes si está presente.

Itera sobre todos los módulos y aplica la inicialización adecuada según el tipo de módulo.
batch_norm: Aplica la normalización por lotes a las entradas.

Toma las entradas, las reestructura y aplica la capa de normalización por lotes.
forward: Define el paso de forward propagation de la red.

Toma las entradas, las pasa a través del modelo secuencial.
Aplica la normalización por lotes si está presente.
Retorna las salidas transformadas.
Utilidad
La clase MLP es útil como un bloque de construcción en muchos modelos de aprendizaje profundo. Se puede utilizar en una variedad de contextos, desde redes neuronales básicas hasta componentes en modelos más complejos,
como codificadores y decodificadores en arquitecturas de aprendizaje profundo. La combinación de capas lineales, activaciones ELU, dropout y normalización por lotes permite a la red aprender representaciones complejas mientras se regula para prevenir el sobreajuste."""
