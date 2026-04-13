import os
import json
import matplotlib.pyplot as plt
import re
import argparse

def extract_epoch_number(folder_name):
    """Wyciąga numer epoki z nazwy folderu (np. 'epoch_10' -> 10)."""
    match = re.search(r'epoch_(\d+)', folder_name)
    return int(match.group(1)) if match else None

def main():
    parser = argparse.ArgumentParser(description="Plot survival metrics across epochs")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the directory containing epoch_ folders")
    parser.add_argument("--output_name", type=str, default="survival_trend.png", help="Name of the output plot file")
    args = parser.parse_args()

    epochs = []
    c_index_avg = []
    c_index_ipcw_avg = []

    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} does not exist.")
        return

    # Pobieranie folderów i sortowanie ich po numerze epoki
    folder_names = [f for f in os.listdir(args.results_dir) if f.startswith('epoch_') and os.path.isdir(os.path.join(args.results_dir, f))]
    folder_names.sort(key=extract_epoch_number)

    if not folder_names:
        print(f"No 'epoch_' folders found in {args.results_dir}")
        return

    print(f"Found {len(folder_names)} epochs. Collecting data...")

    for folder in folder_names:
        epoch_num = extract_epoch_number(folder)
        # Uwzględniam to dziwne podwójne rozszerzenie z Twojego opisu
        file_path = os.path.join(args.results_dir, folder, "Final_results_TCGA-BRCA.star_tpm.json.json")
        
        if not os.path.exists(file_path):
            # Próba znalezienia pliku z pojedynczym .json jeśli podwójne to błąd
            alt_path = os.path.join(args.results_dir, folder, "Final_results_TCGA-BRCA.star_tpm.json")
            file_path = alt_path if os.path.exists(alt_path) else file_path

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    epochs.append(epoch_num)
                    c_index_avg.append(data["Summary"]["c_index"]["avg"])
                    c_index_ipcw_avg.append(data["Summary"]["c_index_ipcw"]["avg"])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing {file_path}: {e}")
        else:
            print(f"Skipping {folder}: File not found.")

    # Generowanie wykresu
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, c_index_avg, marker='o', linewidth=2, label='C-Index (Avg)')
    plt.plot(epochs, c_index_ipcw_avg, marker='s', linestyle='--', linewidth=2, label='C-Index IPCW (Avg)')

    plt.title('Survival Prediction Quality vs. scGPT Adapter Training', fontsize=14)
    plt.xlabel('Epochs of scGPT Adapter Training', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.xticks(epochs)  # Wymuszenie pokazania każdego numeru epoki
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    
    output_path = os.path.join(args.results_dir, args.output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    main()