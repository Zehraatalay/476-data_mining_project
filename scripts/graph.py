import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV yolu
csv_path = "/Users/zehraatalay/Desktop/476-data_mining_project/outputs/models/lstm/validation_training_history.csv"

# Output klasörü (rapor klasörüne koymanı öneriyorum)
output_dir = "/Users/zehraatalay/Desktop/476-data_mining_project/project_report/figures"

# Klasör yoksa oluştur
os.makedirs(output_dir, exist_ok=True)

# CSV oku
df = pd.read_csv(csv_path)

# Plot
plt.figure(figsize=(8, 5))

plt.plot(df["loss"], label="Train Loss")
if "val_loss" in df.columns:
    plt.plot(df["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Kaydet
output_path = os.path.join(output_dir, "lstm_training_curve.png")
plt.savefig(output_path)

print(f"Grafik kaydedildi: {output_path}")

plt.show()