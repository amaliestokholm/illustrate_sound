import subprocess
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import lightkurve as lk


@dataclass
class RecordingSettings:
    sample_rate: int = 24000
    seconds: float = 0.2

    @property
    def format_parec(self) -> str:
        return "float32ne"

    @property
    def format_np(self) -> type:
        return np.float32

    @property
    def sample_width(self) -> int:
        return 4

    @property
    def nchannels(self) -> int:
        return 1

    @property
    def nsamples(self) -> int:
        return int(self.sample_rate * self.seconds)

    @property
    def num_bytes(self) -> int:
        return self.nsamples * self.sample_width * self.nchannels


def audio_stream(settings: RecordingSettings) -> Iterator[np.ndarray]:
    cmdline = [
        "parec",
        f"--latency={settings.num_bytes}",
        f"--channels={settings.nchannels}",
        f"--format={settings.format_parec}",
        "-r",
        "--raw",
        f"--rate={settings.sample_rate}",
    ]
    with subprocess.Popen(
        cmdline, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL
    ) as proc:
        assert proc.stdout is not None
        try:
            while True:
                raw = proc.stdout.read(settings.num_bytes)
                if not raw:
                    break
                data = np.frombuffer(raw, dtype=settings.format_np)
                if data.size == settings.nsamples:
                    yield data
        except GeneratorExit:
            pass
        except KeyboardInterrupt:
            proc.terminate()


def generate_lorentzian_spectrum(
    file="bison_modes.csv", fmin=0.001, fmax=0.006, df=1e-6
):
    modes = pd.read_csv(file, sep=r"\s+", header=None, names=["n", "l", "freq", "error"])
    freqs = np.arange(fmin, fmax, df)
    spectrum = np.zeros_like(freqs)
    width = 100.0
    height = 1.0

    for _, row in modes.iterrows():
        f0 = row["freq"] * 1e-6
        gamma = row["error"] * 1e-6
        H = height
        L = H / (1 + (2 * (freqs - f0) / gamma) ** 2)
        spectrum += L

    return freqs, spectrum


def load_solar_spectrum():
    try:
        freq, power = np.loadtxt("solar_power_spectrum.txt", unpack=True)
    except Exception:
        # From http://bison.ph.bham.ac.uk/portal/frequencies
        # Davies et al. 2014, Hale et al. 2016
        freq, power = generate_lorentzian_spectrum("bison_modes.csv")
    return freq, power


def load_reference_spectrum(star='16 Cyg A', starfile: str | None = None):
    if starfile is not None:
        # Load from saved npz file
        data = np.load(starfile)
        freqs = data['freqs']
        power = data['power']
    else:
        search_result = lk.search_lightcurve(star, mission='Kepler', cadence='short')
        lc = search_result.download_all().stitch()
        pg = lc.to_periodogram(normalization='psd',
                               minimum_frequency=1000,
                               maximum_frequency=2700)
        freqs = pg.frequency.value
        power = pg.power.value
        np.savez_compressed('reference_power.npz', freqs=freqs, power=power)

    return freqs, power


def setup_plot(settings: RecordingSettings):
    fig, (ax, ax1, ax2) = plt.subplots(3, figsize=(10, 8))
    x = np.linspace(0, settings.seconds, settings.nsamples)
    freqs = np.fft.rfftfreq(settings.nsamples, d=1 / settings.sample_rate)

    (line,) = ax.plot(x, np.zeros_like(x), color="dodgerblue")
    (line_fft,) = ax1.plot(freqs, np.zeros_like(freqs), color="orange")

    reference_star = '16 Cyg A'
    if reference_star == 'Sun':
        solar_freq, solar_power = load_solar_spectrum()
        ax2.plot(solar_freq, solar_power, color="orangered")
    else:
        freqs, power = load_reference_spectrum(star=reference_star)
        ax2.plot(freqs, power, color="orangered", label=reference_star)

    ax.set_ylim(-0.4, 0.4)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.3)

    #ax1.set_xscale("log")
    #ax1.set_xlim(20, settings.sample_rate / 2)
    ax1.set_xlim(20, 2000)
    ax1.set_ylim(1e-6, 0.02)
    ax1.set_title("Frequency Spectrum")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.set_title("Solar Power Spectrum (Static)")
    #ax2.set_xlim(0.001, 0.006)  # 1–6 mHz
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.ion()
    fig.show()

    return fig, ax, ax1, line, line_fft, freqs


def update_plot(line, line_fft, freqs, ax1, audio_data):
    line.set_ydata(audio_data)

    fft = np.fft.rfft(audio_data)
    magnitude = np.abs(fft) / len(audio_data)
    line_fft.set_ydata(magnitude)

    peaks, _ = find_peaks(magnitude, prominence=0.01)
    for line in ax1.lines[1:]:  # line_fft is at index 0
        line.remove()
    for text in ax1.texts:
        text.remove()
    ax1.plot(freqs[peaks], magnitude[peaks], "ro", markersize=3)

    for i in range(1, len(peaks)):
        f_diff = freqs[peaks[i]] - freqs[peaks[i - 1]]
        ax1.annotate(
            f"{f_diff:.1f} Hz",
            xy=(freqs[peaks[i]], magnitude[peaks[i]]),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=6,
            color="gray",
            ha="center",
        )

    plt.pause(0.001)


def main():
    settings = RecordingSettings()
    stream = audio_stream(settings)
    fig, ax, ax1, line, line_fft, freqs = setup_plot(settings)

    try:
        for audio in stream:
            update_plot(line, line_fft, freqs, ax1, audio)
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
