import os
import numpy as np
import pandas as pd
from scipy.signal import welch, convolve, windows
from scipy.fftpack import dct
from scipy.stats import entropy
from pathlib import Path

def cortar_segmento(df, start, end):
    """ Corta el DataFrame df entre los tiempos start y end, y devuelve los valores de la columna 'val'."""
    
    return df[(df["time"] >= start) & (df["time"] <= end)]["val"].values


def fft_features(x, fs):
    """ Calcula características FFT de la señal x con frecuencia de muestreo fs."""
    
    if len(x) < fs:
        return {k: np.nan for k in ["fft_dom", "fft_energy", "fft_centroid", "fft_entropy"]}
    
    freqs, psd = welch(x, fs)
    
    dom = freqs[np.argmax(psd)]
    energy = np.sum(psd)
    centroid = np.sum(freqs * psd) / np.sum(psd)
    P = psd / np.sum(psd)
    spec_entropy = entropy(P)
    
    return {
        "fft_dom": dom,
        "fft_energy": energy,
        "fft_centroid": centroid,
        "fft_entropy": spec_entropy
    }

def conv_features(x):
    """ Calcula características de convolución de la señal x."""
    
    if len(x) < 10:
        return {k: np.nan for k in ["conv_d1", "conv_d2", "conv_gauss"]}

    d1 = np.array([1, -1])
    d2 = np.array([1, -2, 1])
    g = windows.gaussian(51, std=7)

    c1 = convolve(x, d1, mode="same")
    c2 = convolve(x, d2, mode="same")
    cg = convolve(x, g, mode="same")

    return {
        "conv_d1": np.sum(c1**2),
        "conv_d2": np.sum(c2**2),
        "conv_gauss": np.sum(cg**2)
    }

def lct_features(x):
    """ Calcula características LCT (DCT) de la señal x."""
    
    if len(x) < 10:
        return {k: np.nan for k in ["lct_mean", "lct_std", "lct_energy"]}

    c = np.abs(dct(x, norm='ortho'))
    return {
        "lct_mean": np.mean(c),
        "lct_std": np.std(c),
        "lct_energy": np.sum(c**2)
    }

def cargar_senales_sujeto(condition, subject_id):
    """Carga todas las señales de un sujeto específico"""

    BASE_DIR = Path(".." ).resolve()
    DATA_DIR = BASE_DIR / "data"

    env_override = os.environ.get("WEARABLE_DATASET_DIR")
    if env_override:
        dataset_root = Path(env_override).expanduser().resolve()
    else:
        direct_path = DATA_DIR / "Wearable_Dataset"
        if direct_path.is_dir():
            dataset_root = direct_path
        else:
            nested_options = list(DATA_DIR.glob("wearable-device-dataset-*/Wearable_Dataset"))
            dataset_root = nested_options[0] if nested_options else None

    sujeto_dir = dataset_root / condition / subject_id
    
    if not sujeto_dir.is_dir():
        print(f"No existe el directorio para {condition}/{subject_id} en {dataset_root}")
        return None
    
    try:
        eda_path = sujeto_dir / "EDA.csv"
        eda_raw = pd.read_csv(eda_path, header=None)
        eda_start = pd.to_datetime(eda_raw.iloc[0,0]).timestamp()
        eda_freq = float(str(eda_raw.iloc[1,0]).strip())
        eda_signal = eda_raw.iloc[2:,0].astype(float).values
        eda_time = eda_start + np.arange(len(eda_signal)) / eda_freq
        EDA = pd.DataFrame({"time": eda_time, "val": eda_signal})
        
        bvp_path = sujeto_dir / "BVP.csv"
        bvp_raw = pd.read_csv(bvp_path, header=None)
        bvp_start = pd.to_datetime(bvp_raw.iloc[0,0]).timestamp()
        bvp_freq = float(str(bvp_raw.iloc[1,0]).strip())
        bvp_signal = bvp_raw.iloc[2:,0].astype(float).values
        bvp_time = bvp_start + np.arange(len(bvp_signal)) / bvp_freq
        BVP = pd.DataFrame({"time": bvp_time, "val": bvp_signal})
        
        temp_path = sujeto_dir / "TEMP.csv"
        temp_raw = pd.read_csv(temp_path, header=None)
        temp_start = pd.to_datetime(temp_raw.iloc[0,0]).timestamp()
        temp_freq = float(str(temp_raw.iloc[1,0]).strip())
        temp_signal = temp_raw.iloc[2:,0].astype(float).values
        temp_time = temp_start + np.arange(len(temp_signal)) / temp_freq
        TEMP = pd.DataFrame({"time": temp_time, "val": temp_signal})
        
        acc_path = sujeto_dir / "ACC.csv"
        acc_raw = pd.read_csv(acc_path, header=None)
        acc_start = pd.to_datetime(acc_raw.iloc[0,0]).timestamp()
        acc_freq = float(str(acc_raw.iloc[1,0]).strip())
        acc_vals = acc_raw.iloc[2:].astype(float).values
        acc_mag = np.sqrt(acc_vals[:,0]**2 + acc_vals[:,1]**2 + acc_vals[:,2]**2)
        acc_time = acc_start + np.arange(len(acc_mag)) / acc_freq
        ACC = pd.DataFrame({"time": acc_time, "val": acc_mag})
        
        tags_path = sujeto_dir / "tags.csv"
        tags_raw = pd.read_csv(tags_path, header=None)
        tags_unix = pd.to_datetime(tags_raw.iloc[:,0]).astype(int) / 1e9
        tags = tags_unix.values
        
        return {
            'EDA': EDA, 'BVP': BVP, 'TEMP': TEMP, 'ACC': ACC,
            'tags': tags,
            'eda_freq': eda_freq, 'bvp_freq': bvp_freq, 
            'temp_freq': temp_freq, 'acc_freq': acc_freq
        }
    except Exception as e:
        print(f"Error cargando {condition}/{subject_id}: {e}")
        return None
    
def generar_segmentos_ventana(tags, window_size=60, overlap=0):
    """Genera segmentos con ventanas deslizantes entre tags consecutivos"""

    segmentos = []
    step = window_size - overlap
    
    for i in range(len(tags) - 1):
        start_phase = tags[i]
        end_phase = tags[i + 1]
        duration = end_phase - start_phase
        
        n_windows = int((duration - window_size) / step) + 1
        
        for w in range(n_windows):
            start = start_phase + w * step
            end = start + window_size
            
            if end <= end_phase:
                segmentos.append({
                    'phase_index': i,
                    'window_index': w,
                    'start_time': start,
                    'end_time': end
                })
    
    return segmentos