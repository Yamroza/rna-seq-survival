"""
run_train.py
------------
Entry point do treningu MLP-A.

Uzycie:
    python run_train.py \
        --sc_data /path/to/single_cell.h5ad \
        --model_dir /path/to/scgpt/whole_human \
        --gene_info_path /path/to/gene_info_table.csv \
        --output_dir ./mlp_a_output \
        --n_epochs 50

Debug (maly dataset, 3 epoki):
    python run_train.py \
        --sc_data /path/to/single_cell.h5ad \
        --model_dir /path/to/scgpt/whole_human \
        --gene_info_path /path/to/gene_info_table.csv \
        --output_dir ./mlp_a_debug \
        --n_epochs 3 \
        --debug_n 500
"""

import argparse
from train import train_mlp_a


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP-A (Path-GPTOmic adapter)")

    # Dane
    parser.add_argument("--sc_data", type=str, required=True,
                        help="Sciezka do danych single-cell (.h5ad)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Sciezka do wag scGPT (np. save/whole_human)")
    parser.add_argument("--gene_info_path", type=str, default=None,
                        help="Sciezka do gene_info_table.csv (mapowanie Ensembl → symbol)")
    parser.add_argument("--cell_type_col", type=str, default="cell_type",
                        help="Kolumna z typem komorki w adata.obs (domyslnie: cell_type)")
    parser.add_argument("--gene_col", type=str, default="gene_name",
                        help="Kolumna z nazwa genu w adata.var (domyslnie: gene_name)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./mlp_a_output",
                        help="Gdzie zapisac wagi i wykresy")

    # Trening
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Rozmiar ukrytych warstw MLP-A (paper: 128)")
    parser.add_argument("--no_precompute", action="store_true",
                        help="Nie pre-komputuj embeddingow scGPT (oszczednosc RAM, wolniejsze)")

    # Debug
    parser.add_argument("--debug_n", type=int, default=None,
                        help="Uzyj tylko pierwszych N komorek (test pipeline)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model, history = train_mlp_a(
        sc_data_path=args.sc_data,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        cell_type_col=args.cell_type_col,
        gene_col=args.gene_col,
        gene_info_path=args.gene_info_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        precompute_embeddings=not args.no_precompute,
        debug_n=args.debug_n,
    )

    print(f"\nGotowe! Wagi w: {args.output_dir}/mlp_a_best.pt")
    print("""
Aby zembedowac bulk RNA-seq:

    from inference import embed_bulk_with_mlp_a
    import scanpy as sc

    bulk = sc.read_h5ad("path/to/bulk.h5ad")
    df = embed_bulk_with_mlp_a(
        bulk,
        model_dir="path/to/scgpt/whole_human",
        mlp_a_weights="./mlp_a_output/mlp_a_best.pt",
        label_encoder_path="./mlp_a_output/label_encoder.pkl",
        output_dir="./mlp_a_output",
        n_random_runs=20,
    )
""")
