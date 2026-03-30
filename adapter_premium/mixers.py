import random
import torch
import numpy as np

class BaseMixer:
    """Klasa bazowa dla wszystkich metod mieszania."""
    def __call__(self, idx, dataset):
        raise NotImplementedError


class NoMixer(BaseMixer):
    def __call__(self, idx, dataset):
        row = dataset.X[idx]
        labels = torch.tensor([dataset.labels[idx]], dtype=torch.long)
        lambdas = torch.tensor([1.0], dtype=torch.float32)
        return row, labels, lambdas


class LinearTwoCellMixer(BaseMixer):
    def __call__(self, idx, dataset):
        random_idx = random.randint(0, dataset.samples - 1)
        lambda_val = random.random()
        
        row_mixed = lambda_val * dataset.X[idx] + (1 - lambda_val) * dataset.X[random_idx]
        
        # Zwracamy dwie etykiety i DWIE wagi
        labels = torch.tensor([dataset.labels[idx], dataset.labels[random_idx]], dtype=torch.long)
        lambdas = torch.tensor([lambda_val, 1 - lambda_val], dtype=torch.float32)
        
        return row_mixed, labels, lambdas


class MultiCellMixer(BaseMixer):
    """Miesza 3 losowe komórki w proporcjach z rozkładu Dirichleta."""
    def __call__(self, idx, dataset):
        # 1. Losujemy dwa dodatkowe indeksy
        idx2 = random.randint(0, dataset.samples - 1)
        idx3 = random.randint(0, dataset.samples - 1)
        
        # 2. Losujemy 3 wagi sumujące się do 1.0
        # Rozkład Dirichleta [1, 1, 1] daje równomierne prawdopodobieństwo różnych mieszanek
        lambdas_np = np.random.dirichlet([1, 1, 1]).astype(np.float32)
        
        # 3. Obliczamy zmieszany wiersz (operacje na macierzach rzadkich)
        mixed_row = (lambdas_np[0] * dataset.X[idx] + 
                     lambdas_np[1] * dataset.X[idx2] + 
                     lambdas_np[2] * dataset.X[idx3])
        
        # 4. Przygotowujemy tensory dla etykiet i wag
        labels = torch.tensor([
            dataset.labels[idx], 
            dataset.labels[idx2], 
            dataset.labels[idx3]
        ], dtype=torch.long)
        
        # Kluczowe: zwracamy lambdy jako tensor PyTorch, aby collate_fn mógł go przetworzyć
        lambdas = torch.from_numpy(lambdas_np)
        
        return mixed_row, labels, lambdas