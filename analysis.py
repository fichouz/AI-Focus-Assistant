import matplotlib.pyplot as plt
import numpy as np
from parser import decode_file 

def sestavi_podatke(seznam_paketov):
    all_data = []
    timestamps = []

    for p in seznam_paketov:
        all_data.append(p.data)
        timestamps.append(p.ts)

    if len(all_data) == 0:
        return 0, np.array([])

    signal = np.vstack(all_data)
    timestamps = np.array(timestamps)
    dt = np.diff(timestamps) / 1000.0  

    Tavg = np.mean(dt)
    Nvz = np.mean([len(p.data) for p in seznam_paketov])
    Fvz = Nvz / Tavg if Tavg > 0 else 0

    return Fvz, signal

def prikazi_signal(signal, Fvz, naslov="", ylabel="", startInd=None, endInd=None):
    if len(signal) == 0:
        print("Ni podatkov.")
        return

    if startInd is not None and endInd is not None:
        signal = signal[startInd:endInd]

    t = np.arange(len(signal)) / Fvz
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal[:, 0], label='x')
    plt.plot(t, signal[:, 1], label='y')
    plt.plot(t, signal[:, 2], label='z')
    plt.title(f"{naslov} (Fvz = {Fvz:.2f} Hz)")
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    file_path = input("Enter the .bin file to analyze (e.g. 1.bin or C:/data/test.bin): ")

    try:
        paketi = decode_file(file_path)
    except Exception as e:
        print(f"Error while opening file: {e}")
        return

    gyro_packets  = [p for p in paketi if p.id == 1]
    accel_packets = [p for p in paketi if p.id == 2]
    mag_packets   = [p for p in paketi if p.id == 3]

    Fg, gyro = sestavi_podatke(gyro_packets)
    Fa, accel = sestavi_podatke(accel_packets)
    Fm, mag = sestavi_podatke(mag_packets)

    gyro = gyro * 8.75e-3
    accel = accel * 6.125e-5
    mag = mag * 1.5e-3

    print("Gyro Fvz:", Fg)
    print("Accel Fvz:", Fa)
    print("Mag Fvz:", Fm)

    prikazi_signal(gyro, Fg, "Gyroscope", "deg/s")
    prikazi_signal(accel, Fa, "Accelerometer", "g")
    prikazi_signal(mag, Fm, "Magnetometer", "Gauss")

    start = int(len(gyro) * 0.3)
    end = start + int(2 * Fg)

    prikazi_signal(
        gyro,
        Fg,
        "Gyroscope (zoom 2s)",
        "deg/s",
        start,
        end
    )

    t_gyro = np.arange(len(gyro)) / Fg
    t_accel = np.arange(len(accel)) / Fa
    t_mag = np.arange(len(mag)) / Fm

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(t_gyro, gyro[:, 0], label='x')
    axs[0].plot(t_gyro, gyro[:, 1], label='y')
    axs[0].plot(t_gyro, gyro[:, 2], label='z')
    axs[0].set_title(
        "Gyroscope\n(configured sampling rate 100 Hz, resolution 8.75e-3 D/s)"
    )
    axs[0].set_ylabel("deg/s")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t_accel, accel[:, 0], label='x')
    axs[1].plot(t_accel, accel[:, 1], label='y')
    axs[1].plot(t_accel, accel[:, 2], label='z')
    axs[1].set_title(
        "Accelerometer\n(configured sampling rate 25 Hz, resolution 6.125e-5 g-force)"
    )
    axs[1].set_ylabel("g")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t_mag, mag[:, 0], label='x')
    axs[2].plot(t_mag, mag[:, 1], label='y')
    axs[2].plot(t_mag, mag[:, 2], label='z')
    axs[2].set_title(
        "Magnetometer\n(configured sampling rate 10 Hz, resolution 1.5e-3 Gauss)"
    )
    axs[2].set_ylabel("Gauss")
    axs[2].set_xlabel("time (s)")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()