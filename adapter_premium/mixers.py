from collections import defaultdict
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


class DynamicDonorMixer(BaseMixer):
    """
    Miesza n_cells komórek, ale tylko w obrębie tego samego dawcy (donor_id).
    Liczbę komórek można podać przy inicjalizacji.
    """
    
    def __init__(self, adata, donor_col="donor_id", tissue_col="tissue_general", n_cells=8):
        super().__init__()
        self.donor_col = donor_col
        self.tissue_col = tissue_col
        self.n_cells = n_cells
        
        # Pobieramy listę dawców
        self.cell_to_donor = adata.obs[donor_col].values
        self.cell_to_tissue = adata.obs[tissue_col].values
        
        # Budujemy mapowanie: (donor, tissue) -> lista indeksów komórek
        self.group_to_indices = defaultdict(list)
        # zip pozwala nam iterować po obu listach jednocześnie
        for idx, (donor, tissue) in enumerate(zip(self.cell_to_donor, self.cell_to_tissue)):
            self.group_to_indices[(donor, tissue)].append(idx)
            
        # Konwertujemy na tablice numpy dla błyskawicznego losowania
        for group in self.group_to_indices:
            self.group_to_indices[group] = np.array(self.group_to_indices[group])

    def __call__(self, idx, dataset):
        # 0. Zabezpieczenie dla 1 komórki (bez mieszania)
        if self.n_cells <= 1:
            row = dataset.X[idx]
            labels = torch.tensor([dataset.labels[idx]], dtype=torch.long)
            lambdas = torch.tensor([1.0], dtype=torch.float32)
            return row, labels, lambdas

        # 1. Sprawdzamy dawcę i tkankę dla głównej komórki
        current_donor = self.cell_to_donor[idx]
        current_tissue = self.cell_to_tissue[idx]
        
        # Klucz złożony:
        current_group = (current_donor, current_tissue)
        
        # Pobieramy dostępne indeksy z tej konkretnej podgrupy
        available_indices = self.group_to_indices[current_group]
        
        # 2. Losujemy pozostałe indeksy
        # replace=True wciąż nas chroni, jeśli w danej tkance od danego dawcy jest mało komórek
        additional_indices = np.random.choice(available_indices, size=self.n_cells - 1, replace=True)
        
        # Łączymy w jedną listę
        all_indices = [idx] + additional_indices.tolist()
        
        # 3. Losujemy wagi (Dirichlet)
        lambdas_np = np.random.dirichlet(np.ones(self.n_cells)).astype(np.float32)
        
        # 4. Mieszamy wiersze 
        mixed_row = lambdas_np[0] * dataset.X[all_indices[0]]
        for i in range(1, self.n_cells):
            mixed_row = mixed_row + lambdas_np[i] * dataset.X[all_indices[i]]
            
        # 5. Przygotowujemy spójne tensory do wyjścia
        labels = torch.tensor([dataset.labels[i] for i in all_indices], dtype=torch.long)
        lambdas = torch.from_numpy(lambdas_np)
        
        return mixed_row, labels, lambdas