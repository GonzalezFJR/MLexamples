import os, time, copy
import torch
import pickle as pkl
import torch.nn as nn
import numpy as np

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, save='', supervised=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # Los mejores pesos se deben guardar por separado
    best_loss = np.inf
    best_epoch = 0
    best_acc = 0.0

    # Cada ciclo tiene una fase de entrenamiento, una de validación y una de prueba
    phases = ['train', 'val', 'test']
    
    # Hacer un seguimiento de la evolución de la pérdida durante el entrenamiento
    training_curves = {}
    for phase in phases:
        training_curves[phase+'_loss'] = []
        training_curves[phase+'_acc'] = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()  # Configurar el modelo en el modo de entrenamiento
            else:
                model.eval()   # Configurar el modelo en el modo de evaluación
            running_corrects = 0
            running_loss = 0.0
            epoch_acc = 0.

            # Iterar con los datos - para el autocodificador no nos importan las etiquetas,
            # estamos entrenando al input contra sí mismo
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                targets = inputs.to(device) if not (supervised) else labels.to(device)

                # Poner a 0 los gradientes de los parámetros
                optimizer.zero_grad()

                # Método forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    # Método backward y actualización de los pesos solo si está en la fase de entrenamiento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Estadísticas
                running_loss += loss.item() * inputs.size(0)
                if supervised: running_corrects += torch.sum(predictions == labels.data)
 
            if phase == 'train':
                scheduler.step()

            dataset_sizes = {}
            for k in dataloaders.keys(): dataset_sizes[k] = len(dataloaders[k].dataset)
            epoch_loss = running_loss / dataset_sizes[phase]
            training_curves[phase+'_loss'].append(epoch_loss)
            epoch_acc = running_corrects.double() / dataset_sizes[phase] if supervised else 0.0
            training_curves[phase+'_acc'].append(epoch_acc)

            print(f'{phase:5} Loss: {epoch_loss:.4f}' + (f' Acc: {epoch_acc:.4f}' if supervised else ''))

            # Hacer una copia profunda del modelo si se ha alcanzado la mejor precisión
            if phase == 'val' and (not supervised or epoch_acc > best_acc):
              best_epoch = epoch
              best_loss = epoch_loss
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f} at epoch {best_epoch}')
    if supervised: print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')

    # Cargar los mejores pesos del modelo
    model.load_state_dict(best_model_wts)
    if save != '':
      if os.path.isfile(save):
        os.system('mv %s %s.old'%(save, save))
      torch.save(model, save)
      with open(save+'.curves', 'wb') as f:
        pkl.dump(training_curves, f)
    
    return model, training_curves 
