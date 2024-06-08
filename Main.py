import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Memuat data dari file CSV (pastikan file Student_Performance.csv ada di direktori yang sama)
data = pd.read_csv("Student_Performance.csv")

# Problem 1: Durasi waktu belajar (TB) terhadap nilai ujian (NT)
TB = data["Hours Studied"].values
NT = data["Performance Index"].values

# Metode 1: Model Linear
def linear_regression(X, y):
    n = len(X)
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = sum((X - X_mean) * (y - y_mean))
    denominator = sum((X - X_mean)**2)
    
    b1 = numerator / denominator
    b0 = y_mean - (b1 * X_mean)
    
    return b0, b1

b0_linear, b1_linear = linear_regression(TB, NT)
NT_pred_linear = b0_linear + b1_linear * TB

# Metode 2: Model Pangkat Sederhana
def power_model(X, y):
    X_log = np.log(X)
    y_log = np.log(y)
    
    b0_log, b1_log = linear_regression(X_log, y_log)
    
    a = np.exp(b0_log)
    b = b1_log
    
    return a, b

a_power, b_power = power_model(TB, NT)
NT_pred_power = a_power * TB**b_power

# Menghitung galat RMS
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

rms_linear = rmse(NT, NT_pred_linear)
rms_power = rmse(NT, NT_pred_power)

print(f"RMS Error - Model Linear: {rms_linear}")
print(f"RMS Error - Model Pangkat: {rms_power}")

# Plot hasil regresi
plt.figure(figsize=(14, 7))

# Plot model linear
plt.subplot(1, 2, 1)
plt.scatter(TB, NT, label='Data asli', color='blue')
plt.plot(TB, NT_pred_linear, label='Regresi Linear', color='red')
plt.xlabel('Durasi waktu belajar (TB)')
plt.ylabel('Nilai ujian (NT)')
plt.title('Model Linear')
plt.legend()

# Plot model pangkat sederhana
plt.subplot(1, 2, 2)
plt.scatter(TB, NT, label='Data asli', color='blue')
plt.plot(TB, NT_pred_power, label='Regresi Pangkat', color='red')
plt.xlabel('Durasi waktu belajar (TB)')
plt.ylabel('Nilai ujian (NT)')
plt.title('Model Pangkat')
plt.legend()

plt.tight_layout()
plt.show()

