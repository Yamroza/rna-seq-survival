import os
import json
import torch
import matplotlib.pyplot as plt
import re
import argparse

def extract_epoch_number(folder_name):
    match = re.search(r'epoch_(\d+)', folder_name)
    return int(match.group(1)) if match else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="survival_vs_acc_trend.png")
    # Dodajemy argument na bazową nazwę pliku, żeby była spójna z treningiem
    parser.add_argument("--ckpt_base_name", type=str, default="epoch")
    args = parser.parse_args()

    epochs, c_index_avg, c_index_ipcw_avg, acc_values = [], [], [], []

    folder_names = [f for f in os.listdir(args.results_dir) if f.startswith('epoch_')]
    folder_names.sort(key=extract_epoch_number)

    print(f"🔍 Found {len(folder_names)} epoch folders.")

    for folder in folder_names:
        epoch_num = extract_epoch_number(folder)
        
        # 1. Ścieżka do JSON
        json_path = os.path.join(args.results_dir, folder, "Final_results_TCGA-BRCA.star_tpm.json.json")
        
        # 2. Ścieżka do Checkpointu - musi pasować do: {base_name}_{epoch}.pt
        ckpt_name = f"{args.ckpt_base_name}_{epoch_num}.pt"
        checkpoint_path = os.path.join(args.checkpoints_dir, ckpt_name)

        if os.path.exists(json_path) and os.path.exists(checkpoint_path):
            try:
                # Ładowanie Accuracy
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                val_acc = ckpt.get('val_acc')
                
                # Ładowanie Survival
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if val_acc is not None:
                    epochs.append(epoch_num)
                    acc_values.append(val_acc)
                    c_index_avg.append(data["Summary"]["c_index"]["avg"])
                    c_index_ipcw_avg.append(data["Summary"]["c_index_ipcw"]["avg"])
            except Exception as e:
                print(f"⚠️ Error processing epoch {epoch_num}: {e}")
        else:
            if not os.path.exists(checkpoint_path):
                print(f"❌ Missing checkpoint: {checkpoint_path}")
            if not os.path.exists(json_path):
                print(f"❌ Missing JSON: {json_path}")

    if not acc_values:
        print("⛔ No data collected! Check if --checkpoints_dir and --ckpt_base_name are correct.")
        return

    # --- SEKCJA GENEROWANIA WYKRESU ---
    plt.figure(figsize=(12, 7))
    
    # Ustawienie kolorów z palety Set1 (wyraźne, zróżnicowane)
    colors = plt.cm.Set1(range(3)) 

    # Plotowanie wszystkich metryk na jednej osi Y
    plt.plot(epochs, c_index_avg, color=colors[0], marker='o', markersize=8, 
             linewidth=2.5, label='Survival C-Index (Avg)')
    
    plt.plot(epochs, c_index_ipcw_avg, color=colors[1], marker='s', markersize=8, 
             linestyle='--', linewidth=2.5, label='Survival C-Index IPCW (Avg)')
    
    plt.plot(epochs, acc_values, color=colors[2], marker='D', markersize=8, 
             linewidth=2.5, label='Cell-Type classification Acc')

    # Tytuł i etykiety
    plt.title('Performance Comparison: Survival Prediction vs. Cell-Type Classification', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('scGPT Adapter Training Epochs', fontsize=13)
    plt.ylabel('Metric Score (0.0 - 1.0)', fontsize=13)

    # Dynamiczne limity osi Y dla lepszej widoczności (z marginesem)
    all_vals = c_index_avg + c_index_ipcw_avg + acc_values
    plt.ylim(min(all_vals) * 0.9, min(max(all_vals) * 1.1, 1.05))

    # Konfiguracja osi X - tylko liczby całkowite (epoki)
    plt.xticks(epochs)

    # Legenda i siatka
    plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.gca().set_facecolor('#f9f9f9')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, args.output_name))
    print(f"✅ Success! Plot saved in {args.results_dir}")

if __name__ == "__main__":
    main()