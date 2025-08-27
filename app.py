import numpy as np
import matplotlib.pyplot as plt
import traceback
import matplotlib
import mpld3
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from scipy.signal import hilbert
import random
from scipy import signal, ndimage
import control as ct
import sympy as sp
import io, base64
from lcapy import LTIFilter
from lcapy.ltifilter import Butterworth, Bessel
import cv2
import os
from io import BytesIO
from lcapy import phasor, symbol, voltage, noisevoltage
from lcapy import StateSpace
from sympy import sympify, Matrix
import control as ct
from lcapy import texpr, s
import matplotlib.patches as patches
app = Flask(__name__)  # Since your HTML is in "main/"

app.secret_key = os.getenv("SECRET_KEY", "defaultsecret")

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/pulse_code_mod_and_demod', methods=['GET', 'POST'])
def pcm():
    result_text = ""
    error = None

    if request.method == 'POST':
        try:
            # Read form inputs (matching HTML form)
            fs = float(request.form.get('sampling_frequency', 100))   # Sampling frequency
            fb = float(request.form.get('message_frequency', 25))     # Message frequency
            A = float(request.form.get('amplitude', 1))               # Amplitude
            bits_per_sample = int(request.form.get('bits', 3))        # Bits per sample
            duration = float(request.form.get('duration', 2))         # Duration in seconds

            # Time vector
            t = np.arange(0, duration, 1/fs)
            
            # Message signal
            message = A * np.sin(2 * np.pi * fb * t)

            # Quantization
            q_levels = 2 ** bits_per_sample
            q_min = np.min(message)
            q_max = np.max(message)
            q_step = (q_max - q_min) / (q_levels - 1)
            quantized = np.round((message - q_min) / q_step) * q_step + q_min

            # Encoding
            bit_stream = []
            for q_val in quantized:
                idx = int((q_val - q_min) / q_step)
                bit_stream.append(format(idx, f'0{bits_per_sample}b'))

            # Decoding
            decoded = []
            for bits in bit_stream:
                idx = int(bits, 2)
                decoded.append(idx * q_step + q_min)
            decoded = np.array(decoded)

            # Prepare text output
            result_text = (
                f"Sampling Frequency: {fs} Hz\n"
                f"Message Frequency: {fb} Hz\n"
                f"Amplitude: {A}\n"
                f"Quantization Levels: {q_levels}\n"
                f"Bits per sample: {bits_per_sample}\n\n"
                f"First 20 Quantized Samples: {quantized[:20]}\n\n"
                f"First 20 Encoded Bits: {bit_stream[:20]}\n\n"
                f"First 20 Decoded Samples: {decoded[:20]}\n"
            )

        except Exception as e:
            error = str(e)

    return render_template('pulse_code_mod_and_demod.html',
                           result_text=result_text, error=error)

@app.route('/amplitude_mod_and_demod', methods=['GET', 'POST'])
def amplitude_mod_and_demod():
    plot_html = ""
    error = None

    # Default values
    defaults = {
        "fs": 4096e6,
        "fb": 64e6,
        "A": 2,
        "N_fft": 2048,
        "fc": 900e6
    }

    try:
        if request.method == 'POST':
            fs = float(request.form.get("fs", defaults["fs"]))
            fb = float(request.form.get("fb", defaults["fb"]))
            A = float(request.form.get("A", defaults["A"]))
            N_fft = int(request.form.get("N_fft", defaults["N_fft"]))
            fc = float(request.form.get("fc", defaults["fc"]))
        else:
            fs, fb, A, N_fft, fc = defaults.values()

        # Time axis
        t = np.arange(N_fft) / fs
        freqs = np.fft.fftfreq(N_fft, 1/fs)

        figs = []  # store all matplotlib figures

        # 1. Baseband signal
        g = A * np.cos(2*np.pi*fb*t)
        g_fft_result = np.fft.fft(g, N_fft)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].plot(g[:200])
        axs[0].set_title('Time Domain Baseband Signal')
        axs[0].set_xlabel('Samples')
        axs[0].set_ylabel('Amplitude')
        axs[1].plot(freqs[:N_fft//2]/1e6, np.abs(g_fft_result[:N_fft//2]))
        axs[1].set_title('One Sided FFT of Baseband Signal')
        axs[1].set_xlabel('Frequency (MHz)')
        axs[1].set_ylabel('Magnitude')
        fig.tight_layout()
        figs.append(fig)

        # 2. Carrier signal
        c = np.cos(2*np.pi*fc*t)
        c_fft_result = np.fft.fft(c, N_fft)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].plot(c[:100])
        axs[0].set_title('Time Domain Carrier Signal')
        axs[0].set_xlabel('Samples')
        axs[0].set_ylabel('Amplitude')
        axs[1].plot(freqs[:N_fft//2]/1e6, np.abs(c_fft_result[:N_fft//2]))
        axs[1].set_title('One Sided FFT of Carrier Signal')
        axs[1].set_xlabel('Frequency (MHz)')
        axs[1].set_ylabel('Magnitude')
        fig.tight_layout()
        figs.append(fig)

        # 3. Modulated signal
        s = g * c
        s_fft_result = np.fft.fft(s, N_fft)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].plot(s[:200])
        axs[0].set_title('Time Domain Modulated Signal')
        axs[0].set_xlabel('Samples')
        axs[0].set_ylabel('Amplitude')
        axs[1].plot(freqs[:N_fft//2]/1e6, np.abs(s_fft_result[:N_fft//2]))
        axs[1].set_title('One Sided FFT of Modulated Signal')
        axs[1].set_xlabel('Frequency (MHz)')
        axs[1].set_ylabel('Magnitude')
        fig.tight_layout()
        figs.append(fig)

        # 4. Demodulated signal (unfiltered)
        x = c * s
        x_fft_result = np.fft.fft(x, N_fft)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].plot(x[:200])
        axs[0].set_title('Time Domain Unfiltered Demodulated Signal')
        axs[0].set_xlabel('Samples')
        axs[0].set_ylabel('Amplitude')
        axs[1].plot(freqs[:N_fft//2]/1e6, np.abs(x_fft_result[:N_fft//2]))
        axs[1].set_title('One Sided FFT of Unfiltered Demodulated Signal')
        axs[1].set_xlabel('Frequency (MHz)')
        axs[1].set_ylabel('Magnitude')
        fig.tight_layout()
        figs.append(fig)

        # 5. Low-pass filtered demodulated signal
        f_cutoff = 0.1  # Fraction of fs
        b = 0.08
        N = int(np.ceil((4 / b)))
        if not N % 2:
            N += 1
        n = np.arange(N)
        h = np.sinc(2 * f_cutoff * (n - (N - 1) / 2))
        w = np.blackman(N)
        h *= w
        h /= np.sum(h)
        u = np.convolve(x, h)
        u_fft_result = np.fft.fft(u, N_fft)
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].plot(u[:200])
        axs[0].set_title('Time Domain Lowpass Filtered Demodulated Signal')
        axs[0].set_xlabel('Samples')
        axs[0].set_ylabel('Amplitude')
        axs[1].plot(freqs[:N_fft//2]/1e6, np.abs(u_fft_result[:N_fft//2]))
        axs[1].set_title('One Sided FFT of Lowpass Filtered Demodulated Signal')
        axs[1].set_xlabel('Frequency (MHz)')
        axs[1].set_ylabel('Magnitude')
        fig.tight_layout()
        figs.append(fig)

        # Convert all figs to HTML and concatenate
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"

    except Exception as e:
        error = str(e)

    return render_template('amplitude_mod_and_demod.html',
                           plot_html=plot_html, error=error)


@app.route('/frequency_mod_and_demod', methods=['GET', 'POST'])
def frequency_mod_and_demod():
    plot_html = ""
    error = None

    defaults = {
        "fs": 10000,     # Sampling frequency
        "duration": 0.01, # Short duration for clear plotting
        "fc": 500,       # Carrier frequency
        "kf": 100,       # Frequency sensitivity
        "fm": 50,        # Message frequency
        "Am": 1          # Message amplitude
    }

    try:
        if request.method == 'POST':
            fs = int(request.form.get("fs", defaults["fs"]))
            duration = float(request.form.get("duration", defaults["duration"]))
            fc = float(request.form.get("fc", defaults["fc"]))
            kf = float(request.form.get("kf", defaults["kf"]))
            fm = float(request.form.get("fm", defaults["fm"]))
            Am = float(request.form.get("Am", defaults["Am"]))
        else:
            fs, duration, fc, kf, fm, Am = defaults.values()

        # Time vector
        t = np.arange(0, duration, 1/fs)

        figs = []

        # 1. Message signal
        message = Am * np.sin(2 * np.pi * fm * t)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t[:500], message[:500])
        ax.set_title("Message Signal (time domain)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        figs.append(fig)

        # 2. FM Modulated signal
        integrated_message = np.cumsum(message) / fs
        fm_signal = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integrated_message)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t[:500], fm_signal[:500])
        ax.set_title("FM Modulated Signal (time domain)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        figs.append(fig)

        # 3. FM Demodulated signal
        analytic_signal = hilbert(fm_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        demodulated = np.diff(instantaneous_phase) * fs / (2 * np.pi * kf)
        demodulated = np.append(demodulated, demodulated[-1])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t[:500], demodulated[:500])
        ax.set_title("FM Demodulated Signal (time domain)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        figs.append(fig)

        # Convert all figs to HTML
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"

    except Exception as e:
        error = str(e)

    return render_template('frequency_mod_and_demod.html',
                           plot_html=plot_html, error=error)

@app.route('/pulse_delta_mod_and_demod', methods=['GET', 'POST'])
def delta_mod_and_demod():
    plot_html = ""
    error = None

    # Default parameters
    defaults = {
        "fs": 1000,      # Sampling frequency
        "duration": 0.05, # Duration in seconds
        "fm": 5,         # Message frequency
        "Am": 1,         # Message amplitude
        "delta": 0.1     # Step size
    }

    try:
        # Get inputs from form (or use defaults)
        if request.method == 'POST':
            fs = int(request.form.get("fs", defaults["fs"]))
            duration = float(request.form.get("duration", defaults["duration"]))
            fm = float(request.form.get("fm", defaults["fm"]))
            Am = float(request.form.get("Am", defaults["Am"]))
            delta = float(request.form.get("delta", defaults["delta"]))
        else:
            fs, duration, fm, Am, delta = defaults.values()

        # Time vector
        t = np.arange(0, duration, 1/fs)

        figs = []

        # 1. Message Signal
        message = Am * np.sin(2 * np.pi * fm * t)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, message)
        ax.set_title("Message Signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        figs.append(fig)

        # 2. Delta Modulation
        y = np.zeros(len(t))
        eq = np.zeros(len(t))
        for i in range(1, len(t)):
            if message[i] > y[i-1]:
                eq[i] = delta
            else:
                eq[i] = -delta
            y[i] = y[i-1] + eq[i]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, message, label="Message")
        ax.step(t, y, label="Delta Modulated", where='post')
        ax.set_title("Delta Modulated Signal")
        ax.legend()
        figs.append(fig)

        # 3. Demodulated Signal (integrating steps)
        demodulated = np.cumsum(eq)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, message, label="Message")
        ax.plot(t, demodulated, label="Demodulated")
        ax.set_title("Delta Demodulated Signal")
        ax.legend()
        figs.append(fig)

        # Convert all figs to HTML
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"

    except Exception as e:
        error = str(e)

    return render_template('pulse_delta_mod_and_demod.html',
                           plot_html=plot_html, error=error)

@app.route('/pulse_width_mod_and_demod', methods=['GET', 'POST'])
def pwm_mod_and_demod():
    plot_html = ""
    error = None

    # Default parameters
    defaults = {
        "fs": 5000,       # Sampling frequency (Hz)
        "f_pwm": 50,      # PWM carrier frequency (Hz)
        "duration": 0.05, # Signal duration (s)
        "seed": 0         # Random seed for reproducibility
    }

    try:
        # Get inputs from form (or defaults)
        if request.method == 'POST':
            fs = int(request.form.get("fs", defaults["fs"]))
            f_pwm = float(request.form.get("f_pwm", defaults["f_pwm"]))
            duration = float(request.form.get("duration", defaults["duration"]))
            seed = int(request.form.get("seed", defaults["seed"]))
        else:
            fs, f_pwm, duration, seed = defaults.values()

        # Time vector
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        # Random waveform
        np.random.seed(seed)
        random_wave = np.random.rand(len(t))  
        random_wave = np.convolve(random_wave, np.ones(50)/50, mode='same')  # smooth

        # Sawtooth carrier for PWM
        carrier = (t * f_pwm) % 1  

        # PWM output
        pwm_output = (random_wave > carrier).astype(float)

        figs = []

        # Plot 1: Random waveform + Carrier
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, random_wave, label='Random Waveform')
        ax.plot(t, carrier, label='Carrier (Sawtooth)', alpha=0.7)
        ax.set_title("Random Waveform vs Carrier")
        ax.legend()
        ax.grid(True)
        figs.append(fig)

        # Plot 2: PWM output
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, pwm_output, drawstyle='steps-pre', label='PWM Output')
        ax.set_title("PWM Signal")
        ax.legend()
        ax.grid(True)
        figs.append(fig)

        # Plot 3: Overlay
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, random_wave, label='Random Waveform')
        ax.plot(t, pwm_output, drawstyle='steps-pre', label='PWM Output')
        ax.set_title("Overlay of Input and PWM")
        ax.legend()
        ax.grid(True)
        figs.append(fig)

        # Convert plots to HTML
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"

    except Exception as e:
        error = str(e)

    return render_template("pulse_width_mod_and_demod.html", plot_html=plot_html, error=error)

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

def huffman_encode(message):
    freq = {}
    for ch in message:
        freq[ch] = freq.get(ch, 0) + 1
    nodes = [Node(ch, f) for ch, f in freq.items()]
    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
        left, right = nodes[0], nodes[1]
        newNode = Node(None, left.freq + right.freq)
        newNode.left, newNode.right = left, right
        nodes = [newNode] + nodes[2:]
    root = nodes[0]

    codes = {}
    def generate_codes(node, code=""):
        if node is None: return
        if node.char is not None: codes[node.char] = code
        generate_codes(node.left, code+"0")
        generate_codes(node.right, code+"1")
    generate_codes(root)

    encoded = "".join(codes[ch] for ch in message)
    return encoded, codes

def huffman_decode(encoded, codes):
    reverse = {v: k for k, v in codes.items()}
    decoded, buff = "", ""
    for bit in encoded:
        buff += bit
        if buff in reverse:
            decoded += reverse[buff]
            buff = ""
    return decoded

# --- RLE ---
def rle_encode(data):
    encoding = ""
    i = 0
    while i < len(data):
        count = 1
        while i + 1 < len(data) and data[i] == data[i+1]:
            i += 1
            count += 1
        encoding += str(count) + data[i]
        i += 1
    return encoding

def rle_decode(data):
    decoded = ""
    count = ""
    for char in data:
        if char.isdigit():
            count += char
        else:
            decoded += char * int(count)
            count = ""
    return decoded

# --- LZW ---
def lzw_compress(uncompressed):
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w, result = "", []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(dictionary[w])
    return result

def lzw_decompress(compressed):
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    result = [dictionary[compressed[0]]]
    w = result[0]
    for k in compressed[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError("Bad LZW compressed k: %s" % k)
        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        w = entry
    return "".join(result)

# --- Add noise ---
def add_noise(data, probability=0.1):
    noisy = ""
    for bit in data:
        if random.random() < probability:
            noisy += "1" if bit == "0" else "0"
        else:
            noisy += bit
    return noisy

@app.route("/channel_coding", methods=["GET", "POST"])
def channel_coding():
    result = None
    error = None

    if request.method == "POST":
        algo = request.form.get("algorithm")
        message = request.form.get("message", "")
        apply_noise = request.form.get("apply_noise")

        if not message:
            error = "Message cannot be empty!"
        else:
            try:
                if algo == "huffman":
                    encoded, codes = huffman_encode(message)
                    if apply_noise:
                        noisy = add_noise(encoded)
                        decoded = huffman_decode(noisy, codes)
                        result = f"Huffman Coding\nMessage: {message}\nEncoded: {encoded}\nNoisy: {noisy}\nDecoded: {decoded}"
                    else:
                        decoded = huffman_decode(encoded, codes)
                        result = f"Huffman Coding\nMessage: {message}\nEncoded: {encoded}\nDecoded: {decoded}"

                elif algo == "rle":
                    encoded = rle_encode(message)
                    decoded = rle_decode(encoded)
                    result = f"RLE\nMessage: {message}\nEncoded: {encoded}\nDecoded: {decoded}"

                elif algo == "lzw":
                    encoded = lzw_compress(message)
                    decoded = lzw_decompress(encoded)
                    result = f"LZW\nMessage: {message}\nEncoded: {encoded}\nDecoded: {decoded}"

                else:
                    error = "Unknown algorithm!"
            except Exception as e:
                error = str(e)

    return render_template("channel_coding.html", result=result, error=error)

@app.route('/complex_QAM', methods=['GET', 'POST'])
def complex_QAM():
    plot_html = ""
    error = None

    # Default values
    defaults = {
        "fs": 4096e6,
        "fb1": 64e6,
        "fb2": 64e6,
        "A1": 2,
        "A2": 1,
        "N_fft": 2048,
        "fc": 900e6
    }

    try:
        if request.method == 'POST':
            fs = float(request.form.get("fs", defaults["fs"]))
            fb1 = float(request.form.get("fb1", defaults["fb1"]))
            fb2 = float(request.form.get("fb2", defaults["fb2"]))
            A1 = float(request.form.get("A1", defaults["A1"]))
            A2 = float(request.form.get("A2", defaults["A2"]))
            N_fft = int(request.form.get("N_fft", defaults["N_fft"]))
            fc = float(request.form.get("fc", defaults["fc"]))
        else:
            fs, fb1, fb2, A1, A2, N_fft, fc = defaults.values()

        # Time axis
        t = np.arange(N_fft) / fs

        # Signals
        g_complex = A1 * np.cos(2*np.pi*fb1*t) + 1j*A2*np.cos(2*np.pi*fb2*t)
        g_fft = np.fft.fft(g_complex, N_fft)
        freqs = np.fft.fftfreq(N_fft, 1/fs)

        c_complex = np.exp(1j*2*np.pi*fc*t)
        v = g_complex * c_complex
        y = v.real

        # Demodulation
        demod_carrier = np.exp(-1j*2*np.pi*fc*t)
        x = y * demod_carrier

        # Lowpass filter
        f_cutoff = 0.1
        b = 0.08
        N = int(np.ceil(4/b))
        if N % 2 == 0: N += 1
        n_filter = np.arange(N)
        h = np.sinc(2*f_cutoff*(n_filter-(N-1)/2)) * np.blackman(N)
        h /= np.sum(h)
        z = np.convolve(x, h)
        z_fft = np.fft.fft(z, N_fft)

        figs = []

        # Helper function
        def add_plot(x_data, y_data, title, xlabel='Samples', ylabel='Amplitude', complex_plot=False):
            fig, ax = plt.subplots(figsize=(8,3))
            if complex_plot:
                ax.plot(x_data, y_data.real)
                ax.plot(x_data, y_data.imag)
                ax.legend(['I Component','Q Component'])
            else:
                ax.plot(x_data, y_data)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.tight_layout()
            figs.append(fig)

        # Generate all plots
        add_plot(t[:200], g_complex[:200], 'Time Domain Complex Information Signal', complex_plot=True)
        add_plot(freqs[:N_fft//2]/1e6, np.abs(g_fft[:N_fft//2]), 'One Sided FFT of Complex Information Signal', 'Frequency MHz', 'Magnitude')
        add_plot(freqs/1e6, np.abs(g_fft), 'Two Sided FFT of Complex Information Signal', 'Frequency MHz', 'Magnitude')

        add_plot(t[:100], c_complex[:100], 'Time Domain Complex Carrier', complex_plot=True)
        add_plot(freqs/1e6, np.abs(np.fft.fft(c_complex,N_fft)), 'FFT of Complex Carrier', 'Frequency MHz', 'Magnitude')

        add_plot(t[:200], v[:200], 'Time Domain Complex Modulated Signal', complex_plot=True)
        add_plot(freqs/1e6, np.abs(np.fft.fft(v,N_fft)), 'FFT of Complex Modulated Signal', 'Frequency MHz', 'Magnitude')

        add_plot(t[:200], y[:200], 'Time Domain Real Modulated Signal')
        add_plot(freqs[:N_fft//2]/1e6, np.abs(np.fft.fft(y,N_fft)[:N_fft//2]), 'One Sided FFT of Real Modulated Signal', 'Frequency MHz', 'Magnitude')

        add_plot(t[:200], x[:200], 'Time Domain Complex Demodulated Signal', complex_plot=True)
        add_plot(freqs/1e6, np.abs(np.fft.fft(x,N_fft)), 'FFT of Complex Demodulated Signal', 'Frequency MHz', 'Magnitude')

        add_plot(t[:200], z[:200], 'Lowpass Filtered Complex Demodulated Signal', complex_plot=True)
        add_plot(freqs[:N_fft//2]/1e6, np.abs(z_fft[:N_fft//2]), 'FFT of Lowpass Filtered Signal', 'Frequency MHz', 'Magnitude')

        # Convert figures to HTML
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"

    except Exception as e:
        error = str(e)

    return render_template('complex_QAM.html', plot_html=plot_html, error=error)

@app.route('/plots')
def show_plots():
    plots = []

    T = 1
    k = 0.25
    A = 1
    border = np.pi

    def plot_to_html(fig):
        return mpld3.fig_to_html(fig)

    def draw_plot1(x_, y_, xlabel, ylabel, title=""):
        fig, ax = plt.subplots()
        ax.stem(x_, y_)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()
        html = plot_to_html(fig)
        plt.close(fig)  # Close figure to avoid memory leaks
        return html

    def c_sum(array, bound):
        csum = []
        s = 0
        for i in range(len(array)):
            if i < bound:
                s += array[i]
            else:
                s = 0
            csum.append(s)
        return csum

    impulse = signal.unit_impulse(100, [0])
    m = np.arange(0, 100, 1)

    # A: simple impulse and step responses
    plots.append(draw_plot1(m, (1 / 2) * (ndimage.shift(impulse, 0, cval=0)[m] + ndimage.shift(impulse, 1, cval=0)[m]),
            'n','Response','Impulse Response'))
    plots.append(draw_plot1(m, np.cumsum((1 / 2) * (ndimage.shift(impulse, 0, cval=0)[m] + ndimage.shift(impulse, 1, cval=0)[m])),
            'n','Response','Step Response'))

    # B: another impulse difference
    plots.append(draw_plot1(m, (ndimage.shift(impulse, -1, cval=0)[m] - ndimage.shift(impulse, 0, cval=0)[m]),
            'n','Response','Impulse Response'))
    plots.append(draw_plot1(m, np.cumsum(ndimage.shift(impulse, -1, cval=0)[m] - ndimage.shift(impulse, 0, cval=0)[m]),
            'n','Response','Step Response'))

    # C: c_sum variations
    for bound in [5, 15, 20]:
        plots.append(draw_plot1(m, c_sum((1 / 2) * (ndimage.shift(impulse, 0, cval=0)[m] + ndimage.shift(impulse, 1, cval=0)[m]), bound),
                               'n', 'Response', f'A Step Response for m_u = {bound}'))
        plots.append(draw_plot1(m, c_sum((ndimage.shift(impulse, -1, cval=0)[m] - ndimage.shift(impulse, 0, cval=0)[m]), bound),
                               'n', 'Response', f'B Step Response for m_u = {bound}'))
        plots.append(draw_plot1(m, c_sum((ndimage.shift(impulse, -1, cval=0)[m] - 
                                          2*ndimage.shift(impulse, 0, cval=0)[m] + 
                                          ndimage.shift(impulse, 1, cval=0)[m]), bound),
                               'n', 'Response', f'C Step Response for m_u = {bound}'))

    return render_template("plots.html", plots=plots)

def vehicle_update(t, x, u, params):
    a = params.get('refoffset', 1.5)
    b = params.get('wheelbase', 3.0)
    maxsteer = params.get('maxsteer', 0.5)
    delta = np.clip(u[1], -maxsteer, maxsteer)
    alpha = np.arctan2(a * np.tan(delta), b)
    return np.array([
        u[0] * np.cos(x[2] + alpha),
        u[0] * np.sin(x[2] + alpha),
        (u[0] / a) * np.sin(alpha)
    ])

def vehicle_output(t, x, u, params):
    return x

@app.route('/control_sys', methods=['GET', 'POST'])
def control_sys():
    plot_html = ""
    error = None

    # Default values
    defaults = {
        "sim_time": 10.0,
        "points": 1000,
        "velocity": 10.0,
        "steer_amp": 0.1,
        "ref_offset": 1.5,
        "wheelbase": 3.0,
        "maxsteer": 0.5
    }

    try:
        if request.method == 'POST':
            sim_time = float(request.form.get("sim_time", defaults["sim_time"]))
            points = int(request.form.get("points", defaults["points"]))
            velocity = float(request.form.get("velocity", defaults["velocity"]))
            steer_amp = float(request.form.get("steer_amp", defaults["steer_amp"]))
            ref_offset = float(request.form.get("ref_offset", defaults["ref_offset"]))
            wheelbase = float(request.form.get("wheelbase", defaults["wheelbase"]))
            maxsteer = float(request.form.get("maxsteer", defaults["maxsteer"]))
        else:
            sim_time, points, velocity, steer_amp, ref_offset, wheelbase, maxsteer = defaults.values()

        # Vehicle system
        vehicle_params = {'refoffset': ref_offset, 'wheelbase': wheelbase, 'velocity': velocity, 'maxsteer': maxsteer}
        vehicle = ct.NonlinearIOSystem(
            vehicle_update, vehicle_output, states=3, name='vehicle',
            inputs=['v','delta'], outputs=['x','y','theta'], params=vehicle_params
        )

        timepts = np.linspace(0, sim_time, points)
        U = [velocity*np.ones_like(timepts), steer_amp*np.sin(timepts*2*np.pi)]

        # Open-loop simulation
        t, outputs = ct.input_output_response(vehicle, timepts, U, 0)
        figs = []

        # 1. Open-loop trajectory
        fig, ax = plt.subplots()
        ax.plot(outputs[0], outputs[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Open-loop Vehicle Trajectory")
        fig.tight_layout()
        figs.append(fig)

        # 2. Inputs over time
        fig, ax = plt.subplots()
        ax.plot(timepts, U[0], 'b-', label='Velocity [m/s]')
        ax.set_ylabel('Velocity [m/s]', color='blue')
        ax2 = ax.twinx()
        ax2.plot(timepts, U[1], 'r-', label='Steering [rad]')
        ax2.set_ylabel('Steering angle [rad]', color='red')
        ax.set_xlabel("Time [s]")
        ax.set_title("Open-loop Inputs")
        fig.tight_layout()
        figs.append(fig)

        # LQR design
        Ud = np.array([velocity*np.ones_like(timepts), np.zeros_like(timepts)])
        Xd = np.array([10*timepts, np.zeros_like(timepts), np.zeros_like(timepts)])
        linsys = vehicle.linearize(Xd[:,0], Ud[:,0])
        K, S, E = ct.lqr(linsys, np.diag([1,1,1]), np.diag([1,1]))
        vehicle_control = ct.NonlinearIOSystem(
            None,
            lambda t, x, z, params={'K':K}: z[3:5] - K@(z[5:8]-z[0:3]),
            name='control',
            inputs=['xd','yd','thetad','vd','deltad','x','y','theta'],
            outputs=['v','delta'],
        )
        vehicle_closed = ct.interconnect(
            (vehicle, vehicle_control),
            inputs=['xd','yd','thetad','vd','deltad'],
            outputs=['x','y','theta']
        )

        # Closed-loop desired trajectory
        Xd = np.array([
            10*timepts + 2*(timepts-5)*(timepts>5),
            0.5*np.sin(timepts*2*np.pi),
            np.zeros_like(timepts)
        ])
        resp = ct.input_output_response(vehicle_closed, timepts, np.vstack((Xd, Ud)), 0)
        time, outputs = resp.time, resp.outputs

        # 3. Closed-loop trajectory comparison
        fig, ax = plt.subplots()
        ax.plot(Xd[0], Xd[1], 'b--', label='Desired Trajectory')
        ax.plot(outputs[0], outputs[1], 'r-', label='Actual Trajectory')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title("Closed-loop Vehicle Trajectory")
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        # 4. Velocity comparison
        fig, ax = plt.subplots()
        ax.plot(np.diff(Xd[0])/np.diff(timepts), 'b--', label='Desired Velocity')
        ax.plot(np.diff(outputs[0])/np.diff(timepts), 'r-', label='Actual Velocity')
        ax.set_xlabel("Time index"); ax.set_ylabel("Velocity [m/s]")
        ax.set_title("Closed-loop Velocity Comparison")
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        # Convert all figs to HTML
        plot_html = "".join([mpld3.fig_to_html(f) + "<br><br>" for f in figs])

    except Exception as e:
        error = str(e)

    return render_template('control_sys.html', plot_html=plot_html, error=error,
                           defaults=defaults)

@app.route("/DSP", methods=["GET", "POST"])
def DSP():
    result = None
    if request.method == "POST":
        try:
            # ----------- Get User Inputs -----------
            x = list(map(int, request.form["x_seq"].split()))
            y = list(map(int, request.form["y_seq"].split()))
            expr_input = request.form["system_expr"]

            x = np.array(x)
            y = np.array(y)

            # ----------- Convolution -----------
            lin_conv = np.convolve(x, y, mode="full")

            N = max(len(x), len(y))
            x_circ = np.pad(x, (0, N - len(x)), mode='constant')
            y_circ = np.pad(y, (0, N - len(y)), mode='constant')
            circ_conv = np.fft.ifft(np.fft.fft(x_circ) * np.fft.fft(y_circ)).real.round().astype(int)

            # ----------- System Expression -----------
            x_sym = sp.Symbol('x')
            system_expr = sp.sympify(expr_input)
            system_func = sp.lambdify(x_sym, system_expr, modules="numpy")

            # ----------- Linearity Check -----------
            x1, x2, a1, a2 = 1, 2, 3, 4  # fixed test values
            y1 = system_func(x1)
            y2 = system_func(x2)
            lhs = system_func(a1 * x1 + a2 * x2)
            rhs = a1 * y1 + a2 * y2
            linearity = "LINEAR ✅" if np.isclose(lhs, rhs) else "NOT LINEAR ❌"

            # ----------- Time Invariance Check -----------
            t0 = 2
            test_vals = [0, 1, 2, 3, 4, 5]
            passed = True
            for val in test_vals:
                y_shifted_input = system_func(val - t0)
                y_shifted_output = system_func(val - t0)
                if not np.isclose(y_shifted_input, y_shifted_output):
                    passed = False
                    break
            time_invariance = "TIME INVARIANT ✅" if passed else "TIME VARYING ❌"

            # ----------- BIBO Stability Check -----------
            bounded_input = np.linspace(-5, 5, 200)
            outputs = system_func(bounded_input)
            bibo = "BIBO STABLE ✅" if np.all(np.abs(outputs) < 1e6) else "NOT BIBO STABLE ❌"

            # ----------- Collect Results -----------
            result = {
                "lin_conv": lin_conv.tolist(),
                "circ_conv": circ_conv.tolist(),
                "system_expr": str(system_expr),
                "linearity": linearity,
                "time_invariance": time_invariance,
                "bibo": bibo
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("DSP.html", result=result)

@app.route("/DFT", methods=["GET", "POST"])
def DFT():
    result = None
    if request.method == "POST":
        try:
            # Get user input
            seq_str = request.form["sequence"]
            x = np.array(list(map(float, seq_str.split())))
            N = len(x)

            # DFT
            X = np.fft.fft(x, N)

            # IDFT
            x_reconstructed = np.fft.ifft(X, N)

            # Prepare results
            dft_list = [f"k={k}: {X[k]:.4f}" for k in range(N)]
            idft_list = [
                f"n={n}: {x_reconstructed[n].real:.4f} + j{x_reconstructed[n].imag:.4e}"
                for n in range(N)
            ]

            result = {
                "sequence": x.tolist(),
                "dft": dft_list,
                "idft": idft_list
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("DFT.html", result=result)

@app.route("/filters", methods=["GET", "POST"])
def filters():
    result = ""
    if request.method == "POST":
        try:
            contin = int(request.form.get("contin"))  # 0=continuous, 1=discrete

            if contin == 1:  # discrete filters
                coeff_mode = int(request.form.get("coeff_mode"))  # 1=coeff, 0=ZPK
                if coeff_mode == 1:
                    num = request.form.get("num", "")
                    den = request.form.get("den", "")
                    b = tuple(map(float, num.strip().split()))
                    a = tuple(map(float, den.strip().split()))
                    fil = LTIFilter(b, a)
                else:
                    zeros = tuple(map(lambda z: complex(z), request.form.get("zeros", "").split()))
                    poles = tuple(map(lambda p: complex(p), request.form.get("poles", "").split()))
                    gain = float(request.form.get("gain", "1"))
                    fil = LTIFilter.from_ZPK(zeros, poles, gain)

                result += f"<pre>{fil.transfer_function()}</pre>"
                result += f"<pre>{fil.frequency_response()}</pre>"
                result += f"<pre>{fil.angular_frequency_response()}</pre>"
                result += f"<pre>{fil.group_delay()}</pre>"
                result += f"<pre>{fil.impulse_response()}</pre>"
                result += f"<pre>{fil.step_response()}</pre>"

            else:  # continuous filters
                filt_type = int(request.form.get("filt_type"))  # 0=Bessel, 1=Butterworth
                omega = float(request.form.get("omega", "1"))

                if filt_type == 1:
                    B = Butterworth(N=2, Wn=omega, btype="lowpass")
                else:
                    B = Bessel(N=2, Wn=omega, btype="lowpass")

                result += f"<pre>{B.transfer_function()}</pre>"
                result += f"<pre>{B.frequency_response()}</pre>"
                result += f"<pre>{B.group_delay()}</pre>"

        except Exception as e:
            result = f"<p style='color:red'>Error: {e}</p>"

    return render_template("filters.html", result=result)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Default image path
DEFAULT_IMAGE = os.path.join(UPLOAD_FOLDER, "1.webp")

def load_image():
    img = cv2.imread(DEFAULT_IMAGE)
    if img is None:
        raise FileNotFoundError("Upload '1.webp' into uploads/ folder.")
    img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@app.route("/image_processing", methods=["GET", "POST"])
def image_processing():
    result_img = None
    operation = None

    if request.method == "POST":
        operation = request.form.get("operation")
        image_rgb = load_image()
        height, width = image_rgb.shape[:2]

        if operation == "scale":
            sx_zoom = float(request.form.get("sx_zoom", 1.5))
            sy_zoom = float(request.form.get("sy_zoom", 1.5))
            zoomed = cv2.resize(image_rgb, (int(width * sx_zoom), int(height * sy_zoom)), interpolation=cv2.INTER_CUBIC)
            result_img = zoomed

        elif operation == "rotate":
            angle = float(request.form.get("angle", 45))
            scale = float(request.form.get("scale", 1.0))
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image_rgb, matrix, (width, height))
            result_img = rotated

        elif operation == "translate":
            tx = int(request.form.get("tx", 50))
            ty = int(request.form.get("ty", 30))
            matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            translated = cv2.warpAffine(image_rgb, matrix, (width, height))
            result_img = translated

        elif operation == "shear":
            shearX = float(request.form.get("shearX", 0.2))
            shearY = float(request.form.get("shearY", 0.2))
            matrix = np.array([[1, shearX, 0], [0, 1, shearY]], dtype=np.float32)
            sheared = cv2.warpAffine(image_rgb, matrix, (width, height))
            result_img = sheared

        elif operation == "normalize":
            b, g, r = cv2.split(image_rgb)
            b = cv2.normalize(b.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
            g = cv2.normalize(g.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
            r = cv2.normalize(r.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
            normalized = cv2.merge((b, g, r))
            result_img = (normalized * 255).astype(np.uint8)

        elif operation == "edges":
            t1 = int(request.form.get("t1", 100))
            t2 = int(request.form.get("t2", 200))
            edges = cv2.Canny(image_rgb, t1, t2)
            result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        elif operation == "blur":
            blurred = cv2.GaussianBlur(image_rgb, (3, 3), 0)
            result_img = blurred

        elif operation == "morphology":
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(gray, kernel, iterations=2)
            result_img = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)

        if result_img is not None:
            # ✅ Return processed image directly (no matplotlib, no white space)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            return send_file(BytesIO(buffer), mimetype="image/png")

    return render_template("image_processing.html")

@app.route('/index1', methods=['GET', 'POST'])
def index1():
    results = []
    error = None

    if request.method == 'POST':
        try:
            kess = int(request.form['kess'])
            dan = bool(kess)

            if dan:  # phasor
                P_val = int(request.form['phasor_val'])
                P = phasor(P_val)
                results.append(f"Time function: {P.time()}")
                results.append(f"Magnitude: {P.magnitude}")
                results.append(f"Phase: {P.phase}")
                results.append(f"RMS: {P.rms()}")
                results.append(f"Omega: {P.omega}")
            else:  # waveform
                expr = request.form['wave_expr']
                V = voltage(phasor(expr))
                results.append(f"Magnitude: {V.magnitude}")
                results.append(f"Phase: {V.phase}")
                results.append(f"RMS: {V.rms()}")
                results.append(f"Omega: {V.omega}")

            # Noise signals
            noise_val = int(request.form['noise_val'])
            X = noisevoltage(noise_val)
            Y = noisevoltage(4)
            Z1 = X + Y
            Z2 = X + Y - X

            results.append("=== Noise Signals observations===")            
            results.append(f"X: units={X.units}, domain={X.domain}, nid={X.nid}, {X}")
            results.append(f"Y: units={Y.units}, domain={Y.domain}, nid={Y.nid}, {Y}")
            results.append(f"Z = X + Y (independent sum): units={Z1.units}, domain={Z1.domain}, nid={Z1.nid}, {Z1}")
            results.append(f"Z = X + Y - X (dependent sum, invalid): units={Z2.units}, domain={Z2.domain}, nid={Z2.nid}, {Z2}")
            results.append("we can observe that X, Y are independent but X+Y, X are not so scalar addition cant be done in noise signals")
        except Exception as e:
            error = str(e)

    return render_template('index1.html', results=results, error=error)

@app.route('/qpsk', methods=['GET', 'POST'])
def qpsk():
    plot_htmls = []
    error = None

    if request.method == 'POST':
        try:
            num_symbols = int(request.form.get('num_symbols', 1000))
            noise_power = float(request.form.get('noise_power', 0.01))
            phase_noise_strength = float(request.form.get('phase_noise', 0.1))

            # QPSK symbols
            x_int = np.random.randint(0, 4, num_symbols)
            x_degrees = x_int * 360 / 4.0 + 45
            x_radians = x_degrees * np.pi / 180.0
            x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)

            fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
            axs = axs.flatten()

            # 1. QPSK without noise
            axs[0].plot(np.real(x_symbols), np.imag(x_symbols), '.')
            axs[0].grid(True)
            axs[0].set_title("QPSK without noise")

            # 2. AWGN noise
            n = (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) / np.sqrt(2)
            r = x_symbols + n * np.sqrt(noise_power)
            axs[1].plot(np.real(r), np.imag(r), '.')
            axs[1].grid(True)
            axs[1].set_title("QPSK with AWGN noise")

            # 3. Phase noise
            phase_noise = np.random.randn(len(x_symbols)) * phase_noise_strength
            r1 = x_symbols * np.exp(1j * phase_noise)
            axs[2].plot(np.real(r1), np.imag(r1), '.')
            axs[2].grid(True)
            axs[2].set_title("QPSK with phase noise")

            # 4. Combination of both noises
            total_noise = r1 + r
            axs[3].plot(np.real(total_noise), np.imag(total_noise), '.')
            axs[3].grid(True)
            axs[3].set_title("QPSK with both phase noise and AWGN noise")

            # Convert figure to HTML base64
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            plot_htmls.append(f'<img src="data:image/png;base64,{img}">')

        except Exception as e:
            error = str(e)

    return render_template('qpsk.html', plots=plot_htmls, error=error)

def matrix_to_float_np(mat):
    return np.array(Matrix(mat).applyfunc(lambda x: float(x.evalf()))).astype(np.float64)

@app.route('/state_space', methods=['GET', 'POST'])
def state_space():
    plots = []
    error = None

    if request.method == 'POST':
        try:
            mode = request.form['mode']
            
            if mode == '0':  # matrices
                A = sympify(request.form['A'])
                B = sympify(request.form['B'])
                C = sympify(request.form['C'])
                D = sympify(request.form['D'])
                ss = StateSpace(A, B, C, D)

            elif mode == '1':  # numerator/denominator
                b = tuple(map(float, request.form['b'].split()))
                a = tuple(map(float, request.form['a'].split()))
                ss = StateSpace.from_transfer_function_coeffs(b, a)

            else:
                raise ValueError("Invalid mode")

            # Convert symbolic matrices to float
            A_np = matrix_to_float_np(ss.A)
            B_np = matrix_to_float_np(ss.B)
            C_np = matrix_to_float_np(ss.C)
            D_np = matrix_to_float_np(ss.D)

            # Control system
            sys = ct.ss(A_np, B_np, C_np, D_np)
            x0 = np.zeros(A_np.shape[0])
            x0[0] = 1
            response = ct.initial_response(sys, T=np.linspace(0, 50, 1000), X0=x0)

            t = response.time
            x = response.states

            # Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            for i in range(x.shape[0]):
                ax.plot(t, x[i], label=f'$x_{i+1}$')
            ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('States')
            ax.set_title('Initial Response')
            ax.grid(True)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            plots.append(img)

        except Exception as e:
            error = str(e)

    return render_template('state_space.html', plots=plots, error=error)

from lcapy import nexpr, z
import traceback

@app.route('/z_transform', methods=['GET', 'POST'])
def z_transform():
    results = []
    error = None

    if request.method == 'POST':
        try:
            user_input = request.form['signal_expr'].strip()
            if user_input == '':
                raise ValueError("Input cannot be empty.")

            expr = nexpr(user_input)
            z_transform = expr(z).simplify()

            results.append({
                'description': 'Original signal',
                'expr': str(expr),
                'latex': expr.latex()
            })

            results.append({
                'description': 'Z-transform (symbolic)',
                'expr': str(z_transform),
                'latex': z_transform.latex()
            })

        except Exception as e:
            error = f"{e}\n{traceback.format_exc()}"

    return render_template('z_transform.html', results=results, error=error)

@app.route('/laplace_transform', methods=['GET', 'POST'])
def laplace_transform():
    results = []
    error = None

    if request.method == 'POST':
        try:
            expr_str = request.form['signal_expr'].strip()
            if expr_str == '':
                raise ValueError("Input cannot be empty.")

            # Parse the input signal
            sig = texpr(expr_str)

            # Compute Laplace transform
            transformed = sig(s).simplify()

            results.append({
                'description': 'Original signal',
                'expr': str(sig),
                'latex': sig.latex()
            })

            results.append({
                'description': 'Laplace transform (symbolic)',
                'expr': str(transformed),
                'latex': transformed.latex()
            })

        except Exception as e:
            error = f"{e}\n{traceback.format_exc()}"

    return render_template('laplace_transform.html', results=results, error=error)

# Descriptions for each plot
PLOT_DESCRIPTIONS = {
    "1": "Pole-Zero Plot: Shows the poles (X) and zeros (O) of the transfer function in the s-plane.",
    "2": "Bode Plot: Displays magnitude (dB) and phase versus frequency, useful for stability and frequency response analysis.",
    "3": "Nyquist Plot: Plots real vs imaginary parts of H(jω), used for stability margins in control systems.",
    "4": "Nichols Plot: Shows open-loop gain (dB) vs phase, combining Bode information in one plot."
}

@app.route("/transfer", methods=["GET", "POST"])
def transfer_page():
    results = []
    error = None

    if request.method == "POST":
        tf_expr = request.form.get("tf_expr", "").strip()
        choices = request.form.getlist("plots")

        if not tf_expr:
            error = "Please enter a transfer function (example: s/(s+2))."
        else:
            try:
                # Safely evaluate the transfer function
                H = eval(tf_expr, {"s": s})

                for choice in choices:
                    plot_obj = None
                    if choice == "1":
                        plot_obj = H.plot()   # Pole-zero
                    elif choice == "2":
                        plot_obj = H.bode_plot()  # Bode
                    elif choice == "3":
                        plot_obj = H.nyquist_plot((-100, 100))  # Nyquist
                    elif choice == "4":
                        plot_obj = H.nichols_plot((-100, 100))  # Nichols

                    if plot_obj is not None:
                        # If it's an Axes, grab its Figure
                        fig = plot_obj.figure if hasattr(plot_obj, "figure") else plot_obj

                        plot_html = mpld3.fig_to_html(fig)
                        results.append({
                            "description": PLOT_DESCRIPTIONS.get(choice, ""),
                            "plot_html": plot_html
                        })

                        plt.close(fig)  # cleanup

            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template("transfer.html", results=results, error=error)
def z_plane(b, a):
    """Generates a Z-plane pole-zero plot and returns it as an mpld3 HTML string."""
    zeros = np.roots(b)
    poles = np.roots(a)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the unit circle
    unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                                 color='black', ls='dashed', alpha=0.5)
    ax.add_patch(unit_circle)

    # Plot zeros and poles
    ax.plot(zeros.real, zeros.imag, 'go', ms=10, markeredgewidth=1.0,
            markeredgecolor='k', markerfacecolor='g', label='Zeros')
    ax.plot(poles.real, poles.imag, 'rx', ms=12, markeredgewidth=3.0,
            markeredgecolor='r', markerfacecolor='r', label='Poles')

    # Axes
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Limits and aspect ratio
    r = 10
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_aspect('equal')

    # Labels & title
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Pole-Zero Plot of Z-Transform')
    plt.legend()

    # Convert figure to mpld3 HTML
    plot_html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return plot_html


@app.route('/z_plane1', methods=['GET', 'POST'])
def z_plane1():
    plot_html = None
    if request.method == 'POST':
        # Get coefficients from user
        numerator = request.form['numerator']
        denominator = request.form['denominator']

        # Convert input (comma or space separated) into list of floats
        b = [float(x) for x in numerator.replace(',', ' ').split()]
        a = [float(x) for x in denominator.replace(',', ' ').split()]

        plot_html = z_plane(b, a)

    return render_template("z_plane1.html", plot_html=plot_html)
def fourier_transfer(b, a):
    """Generates Fourier Transfer Magnitude and Phase plot as an mpld3 HTML string."""
    # Frequency range (0 to π for normalized digital frequencies)
    w = np.linspace(0, np.pi, 1024)
    jw = np.exp(1j * w)

    # Evaluate transfer function H(e^jw) = B(e^jw) / A(e^jw)
    num = np.polyval(b, jw**-1)   # numerator polynomial
    den = np.polyval(a, jw**-1)   # denominator polynomial
    H = num / den

    magnitude = np.abs(H)
    phase = np.angle(H)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))

    # Magnitude plot
    axs[0].plot(w, magnitude, 'b')
    axs[0].set_title("Magnitude Response |H(e^jw)|")
    axs[0].set_xlabel("Frequency (rad/sample)")
    axs[0].set_ylabel("Magnitude")
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # Phase plot
    axs[1].plot(w, phase, 'r')
    axs[1].set_title("Phase Response ∠H(e^jw)")
    axs[1].set_xlabel("Frequency (rad/sample)")
    axs[1].set_ylabel("Phase (radians)")
    axs[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # Convert to mpld3 HTML
    plot_html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return plot_html


@app.route('/fourier_transform_graph', methods=['GET', 'POST'])
def fourier_transfer_route():
    plot_html = None
    if request.method == 'POST':
        # Get coefficients from user
        numerator = request.form['numerator']
        denominator = request.form['denominator']

        # Convert input into list of floats
        b = [float(x) for x in numerator.replace(',', ' ').split()]
        a = [float(x) for x in denominator.replace(',', ' ').split()]

        plot_html = fourier_transfer(b, a)

    return render_template("fourier_transform_graph.html", plot_html=plot_html)

def laplace_transfer(b, a):
    """Generates Laplace Transfer Magnitude and Phase plot as an mpld3 HTML string."""
    # Frequency range for s = jω (continuous-time)
    w = np.linspace(0, 100, 2000)  # rad/sec (adjust upper limit if needed)
    s = 1j * w

    # Evaluate transfer function H(s) = B(s) / A(s)
    num = np.polyval(b, s)
    den = np.polyval(a, s)
    H = num / den

    magnitude = np.abs(H)
    phase = np.angle(H)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))

    # Magnitude plot (log scale, like Bode plot)
    axs[0].semilogx(w, 20 * np.log10(magnitude), 'b')
    axs[0].set_title("Magnitude Response |H(jω)| (dB)")
    axs[0].set_xlabel("Frequency (rad/sec)")
    axs[0].set_ylabel("Magnitude (dB)")
    axs[0].grid(True, which="both", linestyle=':', alpha=0.6)

    # Phase plot
    axs[1].semilogx(w, np.degrees(phase), 'r')
    axs[1].set_title("Phase Response ∠H(jω)")
    axs[1].set_xlabel("Frequency (rad/sec)")
    axs[1].set_ylabel("Phase (degrees)")
    axs[1].grid(True, which="both", linestyle=':', alpha=0.6)

    plt.tight_layout()

    # Convert to mpld3 HTML
    plot_html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return plot_html


@app.route('/laplace_transform_graph', methods=['GET', 'POST'])
def laplace_transfer_route():
    plot_html = None
    if request.method == 'POST':
        # Get coefficients from user
        numerator = request.form['numerator']
        denominator = request.form['denominator']

        # Convert input into list of floats
        b = [float(x) for x in numerator.replace(',', ' ').split()]
        a = [float(x) for x in denominator.replace(',', ' ').split()]

        plot_html = laplace_transfer(b, a)

    return render_template("laplace_transform_graph.html", plot_html=plot_html)

######################################################################


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
