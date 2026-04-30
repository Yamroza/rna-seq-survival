import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

def collect_results(root_dir):
    data = []
    
    # Szukamy ścieżek typu: root_dir/seed_X/limit_size/Final_results*.json
    # Zakładając strukturę: limiting_training_size/seed 1/0.1/Final_results...
    search_path = os.path.join(root_dir, "**", "Final_results*.json")
    files = glob.glob(search_path, recursive=True)

    for file_path in files:
        try:
            # Wyciągamy limit_size z nazwy folderu (np. "0.1")
            path_parts = file_path.split(os.sep)

            # Folder z limitem to zazwyczaj przedostatni element ścieżki
            folder_name = path_parts[-2]
            
            # Regex szukający liczby: opcjonalna cyfra, kropka i cyfry LUB same cyfry
            match = re.search(r"(\d+\.\d+|\d+)", folder_name)
            
            if match:
                limit_size = float(match.group(1))
            else:
                print(f"Nie znaleziono liczby w nazwie folderu: {folder_name}")
                continue
            
            with open(file_path, 'r') as f:
                content = json.load(f)
                
            summary = content.get("Summary", {})
            data.append({
                "limit_size": limit_size,
                "c_index": summary.get("c_index", {}).get("avg"),
                "c_index_ipcw": summary.get("c_index_ipcw", {}).get("avg")
            })
        except (ValueError, KeyError, IndexError) as e:
            print(f"Skipping {file_path} due to error: {e}")

    return pd.DataFrame(data)

# --- Ustawienia ---
root_directory = "../../ts_survival_prediction/src/results/dss_survival_brca/limiting_training_set" # Zmień na właściwą ścieżkę
df = collect_results(root_directory)

# Grupowanie wyników (średnia z różnych seedów dla tego samego limitu)
final_stats = df.groupby("limit_size").mean().sort_index()

print("Zagregowane wyniki:")
print(final_stats)

# --- Wykres ---
plt.figure(figsize=(10, 6))

plt.plot(final_stats.index, final_stats["c_index"], marker='o', linestyle='-', linewidth=2, label='Mean C-Index')
plt.plot(final_stats.index, final_stats["c_index_ipcw"], marker='s', linestyle='--', linewidth=2, label='Mean C-Index IPCW')

plt.title('Model Performance vs Training Subset Size', fontsize=14)
plt.xlabel('Training Subset Size (Fraction)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.xticks(final_stats.index) # Pokaż wszystkie punkty limitów na osi X

# Opcjonalnie: dodanie wartości nad punktami
for x, y in zip(final_stats.index, final_stats["c_index"]):
    plt.text(x, y + 0.005, f'{y:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('limiting_training_size_BRCA.png')