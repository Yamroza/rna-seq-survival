import json
import torch
from pathlib import Path

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained

def load_scgpt_model(
    model_dir: str = "/scratch/2370352/my-research/papers/scgpt/save/whole_human",
    dropout: float = 0.2,
    use_fast_transformer: bool = False,  # True wymaga flash-attn
    pad_token: str = "<pad>",
    pad_value: int = -2,
    device: str = "cpu",
) -> tuple[TransformerModel, GeneVocab]:
    """
    Ładuje pretrenowany model scGPT wraz ze słownikiem genów.
    
    Zwraca:
        model   - TransformerModel z załadowanymi wagami (eval mode, zamrożony)
        vocab   - GeneVocab potrzebny do tokenizacji danych wejściowych
    """
    model_dir = Path(model_dir)
    model_config_file = model_dir / "args.json"
    model_file        = model_dir / "best_model.pt"
    vocab_file        = model_dir / "vocab.json"

    # 1. Wczytaj słownik genów
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # 2. Wczytaj konfigurację architektury z args.json
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize      = model_configs["embsize"]       # np. 512
    nhead        = model_configs["nheads"]        # np. 8
    d_hid        = model_configs["d_hid"]         # np. 512
    nlayers      = model_configs["nlayers"]       # np. 12
    n_layers_cls = model_configs["n_layers_cls"]  # np. 3

    ntokens = len(vocab)

    # 3. Zbuduj model z tą samą architekturą co pretrenowany
    model = TransformerModel(
        ntoken=ntokens,
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,           # bez klasyfikacji w scGPT – adapter zajmie się tym
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        use_fast_transformer=use_fast_transformer,
        pre_norm=False,
    )

    # 4. Załaduj wagi (load_pretrained obsługuje niezgodności kształtów)
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=True)

    model.to(device)
    model.eval()

    # 5. Zamroź wagi scGPT (adapter będzie trenowany, nie backbone)
    for param in model.parameters():
        param.requires_grad = False

    print(f"✓ scGPT załadowany z {model_file} | embsize={embsize}, nlayers={nlayers}")
    return model, vocab
