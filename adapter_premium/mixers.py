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
    
    def __init__(self, adata, donor_col="donor_id", n_cells=8):
        super().__init__()
        self.donor_col = donor_col
        self.n_cells = n_cells
        
        # Pobieramy listę dawców
        self.cell_to_donor = adata.obs[donor_col].values
        
        # Budujemy mapowanie: donor -> lista indeksów komórek
        self.donor_to_indices = defaultdict(list)
        for idx, donor in enumerate(self.cell_to_donor):
            self.donor_to_indices[donor].append(idx)
            
        # Konwertujemy na tablice numpy dla szybkiego losowania
        for donor in self.donor_to_indices:
            self.donor_to_indices[donor] = np.array(self.donor_to_indices[donor])

    def __call__(self, idx, dataset):
        # 0. Zabezpieczenie: Jeśli n_cells ustawiono na 1, po prostu zwracamy komórkę bez zmian
        if self.n_cells <= 1:
            row = dataset.X[idx]
            labels = torch.tensor([dataset.labels[idx]], dtype=torch.long)
            lambdas = torch.tensor([1.0], dtype=torch.float32)
            return row, labels, lambdas

        # 1. Sprawdzamy dawcę dla głównej komórki
        current_donor = self.cell_to_donor[idx]
        available_indices = self.donor_to_indices[current_donor]
        
        # 2. Losujemy pozostałe indeksy (n_cells - 1)
        # replace=True zapobiega błędom, gdy dawca ma mniej komórek niż n_cells (wtedy po prostu dociągnie tę samą komórkę dwa razy)
        additional_indices = np.random.choice(available_indices, size=self.n_cells - 1, replace=True)
        
        # Łączymy główny indeks z wylosowanymi
        all_indices = [idx] + additional_indices.tolist()
        
        # 3. Losujemy wagi sumujące się do 1.0 dla n_cells
        lambdas_np = np.random.dirichlet(np.ones(self.n_cells)).astype(np.float32)
        
        # 4. Mieszamy wiersze (rzadkie macierze scipy.sparse obsługują dodawanie)
        mixed_row = lambdas_np[0] * dataset.X[all_indices[0]]
        for i in range(1, self.n_cells):
            mixed_row = mixed_row + lambdas_np[i] * dataset.X[all_indices[i]]
            
        # 5. Przygotowujemy tensory do zwrócenia
        labels = torch.tensor([dataset.labels[i] for i in all_indices], dtype=torch.long)
        lambdas = torch.from_numpy(lambdas_np)
        
        return mixed_row, labels, lambdas