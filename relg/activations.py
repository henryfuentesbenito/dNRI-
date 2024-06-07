import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.data + y_soft
    else:
        y = y_soft
    return y


"""El código implementa la técnica de Gumbel-Softmax, que se utiliza para muestrear una distribución categórica de manera diferenciable. 
Esta técnica es útil en modelos de aprendizaje profundo donde se necesita hacer una selección discreta pero se requiere que el proceso 
sea diferenciable para poder entrenar usando retropropagación.

Funciones
sample_gumbel

Descripción: Genera muestras de ruido de Gumbel.
Parámetros:
shape: La forma del tensor de salida.
eps: Un pequeño valor para evitar la división por cero.
Funcionamiento: Crea un tensor de números aleatorios uniformes, aplica la transformación de Gumbel, y retorna el ruido de Gumbel.
gumbel_softmax_sample

Descripción: Genera muestras utilizando el muestreo de Gumbel-Softmax.
Parámetros:
logits: Los logits de entrada.
tau: La temperatura de Gumbel-Softmax.
eps: Un pequeño valor para evitar la división por cero.
Funcionamiento: Añade ruido de Gumbel a los logits y aplica la función softmax ajustada con la temperatura dada.
gumbel_softmax

Descripción: Aplica la operación Gumbel-Softmax para muestrear de una distribución categórica de manera diferenciable.
Parámetros:
logits: Los logits de entrada.
tau: La temperatura de Gumbel-Softmax.
hard: Si True, devuelve un vector one-hot.
eps: Un pequeño valor para evitar la división por cero.
Funcionamiento:
Calcula los valores soft usando gumbel_softmax_sample.
Si hard es True, convierte las muestras en vectores one-hot utilizando la función scatter_.
Devuelve los valores hard ajustados con los valores soft para mantener la diferenciabilidad.
Utilidad
El Gumbel-Softmax es útil en situaciones donde necesitamos realizar una selección discreta, pero queremos que el proceso sea diferenciable 
para permitir la retropropagación. Esto es especialmente útil en modelos generativos y en tareas donde se necesita aprender una distribución sobre categorías discretas, como en redes neuronales 
que trabajan con datos secuenciales y relacionales. La capacidad de hacer que la salida sea exactamente one-hot mientras mantiene la diferenciabilidad hace que esta técnica sea poderosa para 
modelos que requieren decisiones discretas durante el entrenamiento."""