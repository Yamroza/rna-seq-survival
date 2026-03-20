# src/model.py

import torch
import torch.nn as nn


class MLPAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class scGPTAdapterModel(nn.Module):
    """
    scGPT (zamrożony) → MLPAdapter → head klasyfikacyjny.

    DataLoader musi zwracać słownik z kluczami:
        "gene_ids"  : LongTensor  [batch, seq_len]   – indeksy tokenów genów
        "values"    : FloatTensor [batch, seq_len]    – zbinowane wartości ekspresji
        "labels"    : LongTensor  [batch]             – klasy komórek
    
    Parametry `pad_token` i `pad_value` muszą być zgodne z tym,
    czego użyto przy ładowaniu modelu w load_scgpt_model().
    """

    def __init__(
        self,
        scgpt_model,
        input_dim: int,          # embsize scGPT, np. 512
        latent_dim: int = 256,
        num_classes: int = 7,
        pad_token: str = "<pad>",
        vocab=None,              # GeneVocab – potrzebna do budowania maski
    ):
        super().__init__()

        self.scgpt = scgpt_model
        self.scgpt.requires_grad_(False)   # backbone zamrożony

        self.pad_token_id = vocab[pad_token] if vocab is not None else 0

        self.adapter = MLPAdapter(
            input_dim=input_dim,
            output_dim=latent_dim,
        )
        self.head = nn.Linear(latent_dim, num_classes)

    def _get_cell_embedding(
        self,
        gene_ids: torch.Tensor,   # [batch, seq_len]
        values: torch.Tensor,     # [batch, seq_len]
    ) -> torch.Tensor:            # [batch, embsize]
        """
        Wywołuje wewnętrzny encoder scGPT i wyciąga embedding CLS
        (pierwszy token w sekwencji = reprezentacja całej komórki).
        """
        # maska True tam, gdzie pad – scGPT ignoruje te pozycje
        src_key_padding_mask = gene_ids.eq(self.pad_token_id)   # [batch, seq_len]

        with torch.no_grad():
            # _encode zwraca [batch, seq_len, embsize]
            transformer_output = self.scgpt._encode(
                gene_ids,
                values,
                src_key_padding_mask,
            )

        # Embedding komórki = token CLS (pozycja 0)
        # Zgodne z cell_emb_style="cls" używanym w scGPT_human
        cell_emb = transformer_output[:, 0, :]   # [batch, embsize]
        return cell_emb

    def forward(
        self,
        gene_ids: torch.Tensor,   # [batch, seq_len]  LongTensor
        values: torch.Tensor,     # [batch, seq_len]  FloatTensor
    ):
        # 1. Embeddingi z zamrożonego scGPT
        cell_emb = self._get_cell_embedding(gene_ids, values)   # [batch, embsize]

        # 2. Adapter dostosowuje przestrzeń latentną
        z = self.adapter(cell_emb)    # [batch, latent_dim]

        # 3. Głowa klasyfikacyjna
        out = self.head(z)            # [batch, num_classes]

        return z, out