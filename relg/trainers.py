import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import tqdm



def build_scheduler(opt, decay_factor=0.1, decay_steps=20):
    return torch.optim.lr_scheduler.StepLR(opt, decay_steps, decay_factor)

class Trainer:
    def __init__(self, model, generator, epochs, optimizer=None, scheduler=None, **kwargs):
        self.model = model
        self.generator = generator
        self.epochs = epochs

        self.set_optimizer(optimizer)
        self.set_scheduler(scheduler)

    def set_optimizer(self, optimizer):
        if optimizer:
            self.opt = optimizer
        else:
            model_params = [param for param in self.model.parameters() if param.requires_grad]
            self.opt = torch.optim.Adam(model_params, lr=5e-4)

    def set_scheduler(self, scheduler):
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = build_scheduler(self.opt, 0.1, 20)

    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.scheduler.step()

    def train_one_epoch(self):
        pbar = tqdm.tqdm(self.generator, total=len(self.generator), desc="Batches", position=1)
        self.model.train()

        for batch in pbar:
            inputs = batch["inputs"]
            # inputs = inputs.cuda(non_blocking=True)
            # inputs = inputs.cpu()
            self.opt.zero_grad()
            loss, nll_loss, kl_loss = self.model.compute_loss(inputs)
            loss.backward()

            pbar.set_postfix(
                {"loss": loss.item(), "nll": nll_loss.mean().item(), "kl": kl_loss.mean().item(), "lr": self.opt.param_groups[0]["lr"]},
                refresh=True,
            )
            self.opt.step()















"""def build_scheduler(opt, decay_factor=0.1, decay_steps=20):
    return torch.optim.lr_scheduler.StepLR(opt, decay_steps, decay_factor)

class Trainer:
    def __init__(self, model, generator, epochs, optimizer=None, scheduler=None, device='cpu', **kwargs):
        self.model = model
        self.generator = generator
        self.epochs = epochs
        self.device = device

        self.set_optimizer(optimizer)
        self.set_scheduler(scheduler)

    def set_optimizer(self, optimizer):
        if optimizer:
            self.opt = optimizer
        else:
            model_params = [param for param in self.model.parameters() if param.requires_grad]
            self.opt = torch.optim.Adam(model_params, lr=5e-4)

    def set_scheduler(self, scheduler):
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = build_scheduler(self.opt, 0.1, 20)

    def train(self):
        for _ in tqdm.tqdm(range(self.epochs), desc="Epochs", position=0):
            pbar = tqdm.tqdm(self.generator, total=len(self.generator), desc="Batches", position=1)
            self.model.train()

            for batch in pbar:
                inputs = batch["inputs"].to(self.device)

                self.opt.zero_grad()
                loss, nll_loss, kl_loss = self.model.compute_loss(inputs)
                loss.backward()

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "nll": nll_loss.mean().item(),
                        "kl": kl_loss.mean().item(),
                        "lr": self.opt.param_groups[0]["lr"],
                    },
                    refresh=True,
                )
                self.opt.step()
            self.scheduler.step()"""








"""def build_scheduler(opt, decay_factor=0.1, decay_steps=20):
    return torch.optim.lr_scheduler.StepLR(opt, decay_steps, decay_factor)


class Trainer:
    def __init__(
        self,
        model,
        generator,
        epochs,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ):
        self.model = model
        self.generator = generator
        self.epochs = epochs

        self.set_optimizer(optimizer)
        self.set_scheduler(scheduler)

    def set_optimizer(self, optimizer):
        if optimizer:
            self.opt = optimizer
        else:
            model_params = [
                param for param in self.model.parameters() if param.requires_grad
            ]
            self.opt = torch.optim.Adam(model_params, lr=5e-4)

    def set_scheduler(self, scheduler):
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = build_scheduler(self.opt, 0.1, 20)

    def train(self):
        for _ in tqdm.tqdm(range(self.epochs), desc="Epochs", position=0):
            pbar = tqdm.tqdm(
                self.generator,
                total=len(self.generator),
                desc="Batches",
                position=1,
            )
            self.model.train()

            for batch in pbar:
                inputs = batch["inputs"]
                inputs = inputs.cuda(non_blocking=True)

                self.opt.zero_grad()
                loss, nll_loss, kl_loss = self.model.compute_loss(inputs)
                loss.backward()

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "nll": nll_loss.mean().item(),
                        "kl": kl_loss.mean().item(),
                        "lr": self.opt.param_groups[0]["lr"],
                    },
                    refresh=True,
                )
                self.opt.step()
            self.scheduler.step()"""



"""La clase Trainer es una implementación de un entrenador para modelos de aprendizaje profundo. Su función principal es orquestar el proceso de entrenamiento del modelo, 
incluyendo la gestión del optimizador, el scheduler (programador de la tasa de aprendizaje), y el bucle de entrenamiento que itera sobre los datos, calcula las pérdidas y ajusta los pesos del modelo.

Atributos
model: El modelo de aprendizaje profundo que se va a entrenar.
generator: Un generador de datos que proporciona los lotes de datos para el entrenamiento.
epochs: El número de épocas (ciclos completos sobre el conjunto de datos) durante las cuales se entrenará el modelo.
opt: El optimizador utilizado para actualizar los pesos del modelo.
scheduler: El scheduler utilizado para ajustar la tasa de aprendizaje durante el entrenamiento.
Métodos
__init__: Inicializa el entrenador con el modelo, el generador de datos, el número de épocas, el optimizador y el scheduler.

Configura el optimizador y el scheduler mediante set_optimizer y set_scheduler.
set_optimizer: Configura el optimizador para el entrenamiento.

Si se proporciona un optimizador, lo utiliza.
Si no, crea un optimizador Adam con una tasa de aprendizaje predeterminada (0.0005) para los parámetros del modelo que requieren gradiente.
set_scheduler: Configura el scheduler para ajustar la tasa de aprendizaje.

Si se proporciona un scheduler, lo utiliza.
Si no, crea un scheduler con un decaimiento de la tasa de aprendizaje del 10% cada 20 pasos.
train: Ejecuta el bucle de entrenamiento del modelo.

Itera sobre el número de épocas especificadas.
Dentro de cada época, itera sobre los lotes de datos proporcionados por el generador.
Realiza las siguientes acciones para cada lote de datos:
Mueve los datos de entrada a la GPU.
Pone el modelo en modo de entrenamiento.
Calcula la pérdida total, la pérdida de verosimilitud negativa (NLL) y la pérdida de divergencia KL utilizando el método compute_loss del modelo.
Realiza la retropropagación de la pérdida.
Actualiza los pesos del modelo mediante el optimizador.
Ajusta la tasa de aprendizaje mediante el scheduler.
Actualiza y muestra las estadísticas de la pérdida y la tasa de aprendizaje en la barra de progreso.
Utilidad
La clase Trainer es útil para automatizar el proceso de entrenamiento de modelos de aprendizaje profundo. Proporciona una estructura organizada 
para manejar el bucle de entrenamiento, incluyendo la gestión del optimizador y el scheduler, así como la actualización de los pesos del modelo. Esta clase facilita la experimentación con diferentes 
configuraciones de modelos y parámetros de entrenamiento, permitiendo un ajuste eficiente y efectivo del modelo a los datos."""