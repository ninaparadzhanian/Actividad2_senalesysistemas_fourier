import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración inicial
fs = 1000  # Frecuencia de muestreo
t = np.linspace(-1, 1, fs, endpoint=False)

# Definición de señales
signals = {
    "Pulso Rectangular": np.where(np.abs(t) <= 0.2, 1.0, 0.0),
    "Función Escalón": np.where(t >= 0, 1.0, 0.0),
    "Señal Senoidal (10Hz)": np.sin(2 * np.pi * 10 * t)
}

# 2. Función para graficar Dominio del Tiempo vs Frecuencia
def plot_fourier_analysis(signals, fs, t):
    fig, axes = plt.subplots(len(signals), 3, figsize=(15, 10))
    fig.tight_layout(pad=4.0)

    for i, (name, sig) in enumerate(signals.items()):
        # --- Dominio del Tiempo ---
        axes[i, 0].plot(t, sig, color='blue')
        axes[i, 0].set_title(f"{name} (Tiempo)")
        axes[i, 0].grid(True)

        # --- Cálculo de FFT ---
        n = len(sig)
        fft_res = np.fft.fftshift(np.fft.fft(sig))
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1/fs))
        
        magnitud = np.abs(fft_res) / n
        fase = np.angle(fft_res)
        fase[magnitud < 0.01] = 0  # Limpiar ruido de fase en magnitudes despreciables

        # --- Magnitud ---
        axes[i, 1].plot(freqs, magnitud, color='red')
        axes[i, i].set_xlim([-50, 50]) # Zoom en el área de interés
        axes[i, 1].set_title("Magnitud del Espectro")
        axes[i, 1].grid(True)

        # --- Fase ---
        axes[i, 2].plot(freqs, fase, color='green')
        axes[i, 2].set_xlim([-50, 50])
        axes[i, 2].set_title("Fase (Radianes)")
        axes[i, 2].grid(True)

    plt.show()

plot_fourier_analysis(signals, fs, t)