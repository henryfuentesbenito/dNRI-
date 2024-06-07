import torch
from torch.utils.data import DataLoader
import relg
import os
import argparse

import torch
from torch.utils.data import DataLoader
import relg
import os
import argparse

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def save_final_model(model, checkpoint_dir):
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

def main(start_epoch=0, total_epochs=100, checkpoint_dir="models"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = relg.dnri(
        input_dim=4,
        encoder_hidden_dim=96,
        rnn_encoder_dim=96,
        decoder_dim=96,
        num_objects=3,
        num_edge_types=2,
        prior=[0.9, 0.1],
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load checkpoint if starting epoch is not 0
    if start_epoch > 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{start_epoch - 1}.pth")
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path) + 1

    dataset = relg.SynthDataLoader("/content/drive/MyDrive/relationalgnns/datasets", "train")
    generator = relg.LoaderWrapper(
        DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            drop_last=True,
        ),
        512,
    )

    trainer = relg.Trainer(model, generator, epochs=1, optimizer=optimizer, device=device)

    for epoch in range(start_epoch, total_epochs):
        trainer.train_one_epoch()
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)
        print(f"Checkpoint saved for epoch {epoch}")

    save_final_model(model, checkpoint_dir)
    print(f"Final model saved at {os.path.join(checkpoint_dir, 'final_model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from')
    parser.add_argument('--total_epochs', type=int, default=100, help='Total number of epochs to train')
    parser.add_argument('--checkpoint_dir', type=str, default='models', help='Directory to save checkpoints')
    args = parser.parse_args()
    main(start_epoch=args.start_epoch, total_epochs=args.total_epochs, checkpoint_dir=args.checkpoint_dir)













"""import torch
from torch.utils.data import DataLoader

import relg


model = relg.dnri(
    input_dim=4,
    encoder_hidden_dim=96,
    rnn_encoder_dim=96,
    decoder_dim=96,
    num_objects=3,
    num_edge_types=2,
    prior=[0.9, 0.1],
)

dataset = relg.SynthDataLoader("datasets", "train")
generator = relg.LoaderWrapper(
    DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
    ),
    512,
)

trainer = relg.Trainer(model, generator, epochs=100)
trainer.train()

torch.save(model.encoder, "models/encoder_09_prior")"""










"""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = relg.dnri(
    input_dim=4,
    encoder_hidden_dim=96,
    rnn_encoder_dim=96,
    decoder_dim=96,
    num_objects=3,
    num_edge_types=2,.\venv\Scripts\Activate.ps1
    prior=[0.9, 0.1],
).to(device)

dataset = relg.SynthDataLoader("datasets", "train")
generator = relg.LoaderWrapper(
    DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
    ),
    512,
)

trainer = relg.Trainer(model, generator, epochs=100)
trainer.train()

torch.save(model.encoder, "models/encoder_09_prior")"""


"""Este código configura y ejecuta el entrenamiento de un modelo dnri (Dynamic Neural Relational Inference) utilizando PyTorch. A continuación se describe cada parte del código en detalle.

1. Importaciones
torch y torch.utils.data: Se utilizan para la creación y manejo de tensores y para la carga de datos.
relg: Un módulo personalizado (no estándar) que se asume contiene las definiciones de las clases y funciones necesarias para el modelo dnri, el cargador de datos SynthDataLoader, 
el envoltorio de cargador LoaderWrapper, y el entrenador Trainer.



input_dim: Dimensión de la entrada.
encoder_hidden_dim: Dimensión oculta para el codificador.
rnn_encoder_dim: Dimensión oculta para la parte recurrente del codificador.
decoder_dim: Dimensión oculta para el decodificador.
num_objects: Número de objetos en el sistema.
num_edge_types: Número de tipos de bordes entre los objetos.
prior: Prior para los tipos de bordes, en este caso, 90% para el primer tipo y 10% para el segundo.

dataset: Instancia de SynthDataLoader que carga los datos de entrenamiento desde el directorio datasets.
generator: Un LoaderWrapper que envuelve un DataLoader de PyTorch.
batch_size=8: Tamaño del lote de datos.
shuffle=True: Baraja los datos en cada época.
drop_last=True: Descarta el último lote si no completa un lote completo.
512: Número de pasos a iterar.


trainer: Instancia de la clase Trainer que toma el modelo y el generador de datos, y se configura para entrenar durante 100 épocas.
trainer.train(): Ejecuta el proceso de entrenamiento.


Guarda el estado del codificador del modelo en un archivo denominado encoder_09_prior en el directorio models.



Este código ilustra un flujo típico de entrenamiento en PyTorch:

Definición y configuración del modelo.
Configuración de los cargadores de datos.
Configuración del entrenador.
Ejecución del entrenamiento.
Guardado del modelo.
El modelo dnri parece estar diseñado para tareas de inferencia relacional dinámica, donde se modelan y predicen relaciones cambiantes entre objetos en sistemas complejos. 
Este tipo de modelo es útil en una variedad de aplicaciones, incluyendo simulaciones físicas, modelado de sistemas biológicos, y análisis de redes sociales, entre otros."""







