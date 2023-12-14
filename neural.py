import numpy as np
import torch
import torch.nn as nn




class MLPptorchReLU(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size):
        super(MLPptorchReLU, self).__init__()

        # Создаем список слоев
        layers_list = []

        # Добавляем входной слой
        layers_list.append(nn.Linear(in_size, hidden_sizes[0]))
        layers_list.append(nn.ReLU())

        # Добавляем скрытые слои
        for i in range(len(hidden_sizes) - 1):
            layers_list.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers_list.append(nn.ReLU())

        # Добавляем выходной слой
        layers_list.append(nn.Linear(hidden_sizes[-1], out_size))
        layers_list.append(nn.ReLU())

        # Инициализируем Sequential
        self.layers = nn.Sequential(*layers_list)

    # Прямой проход
    def forward(self, x):
        return self.layers(x)
    
    
    
class MLPptorchSigmoid(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size):
        super(MLPptorchSigmoid, self).__init__()

        # Создаем список слоев
        layers_list = []

        # Добавляем входной слой
        layers_list.append(nn.Linear(in_size, hidden_sizes[0]))
        layers_list.append(nn.Sigmoid())

        # Добавляем скрытые слои
        for i in range(len(hidden_sizes) - 1):
            layers_list.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers_list.append(nn.Sigmoid())

        # Добавляем выходной слой
        layers_list.append(nn.Linear(hidden_sizes[-1], out_size))
        layers_list.append(nn.Sigmoid())

        # Инициализируем Sequential только теми слоями, которые являются экземплярами nn.Module
        self.layers = nn.ModuleList(layers_list)

    # Прямой проход
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x