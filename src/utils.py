import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_signal(path, colname=None):
    """
    Carga una señal fisiológica en formato Empatica E4 y reconstruye su eje temporal.
    
    Formato Empatica:
    - Línea 1: Timestamp inicial (Unix time o fecha "YYYY-MM-DD HH:MM:SS")
    - Línea 2: Frecuencia de muestreo (Hz)
    - Línea 3+: Valores de la señal (1 columna o 3 para ACC)
    """
    raw = pd.read_csv(path, header=None)
    
    start_str = str(raw.iloc[0, 0]).strip().replace("\ufeff", "")
    
    try:
        start_ts = float(start_str)
    except ValueError:
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            start_ts = start_dt.timestamp()
        except:
            start_dt = datetime.fromisoformat(start_str)
            start_ts = start_dt.timestamp()
    
    raw_freq = str(raw.iloc[1, 0]).strip().replace("\ufeff", "")
    freq = float(raw_freq)
    
    values_raw = raw.iloc[2:].reset_index(drop=True)
    
    if values_raw.shape[1] == 1:
        values = pd.to_numeric(values_raw.iloc[:, 0], errors="coerce").values
    else:
        values = values_raw.apply(pd.to_numeric, errors='coerce').values
    
    n = len(values)
    time = start_ts + np.arange(n) / freq
    
    if colname is not None:
        if values.ndim == 1:
            df = pd.DataFrame({colname: values, 'time': time})
        else:
            df = pd.DataFrame({
                f'{colname}_x': values[:, 0],
                f'{colname}_y': values[:, 1],
                f'{colname}_z': values[:, 2],
                'time': time
            })
        return df, freq
    else:
        return {
            'start_time': start_ts,
            'sampling_rate': freq,
            'values': values,
            'time': time
        }

def load_tags(subject_path):
    """ Carga el archivo tags.csv de un sujeto, manejando múltiples formatos."""    

    if subject_path.endswith('tags.csv'):
        tags_path = subject_path
    else:
        tags_path = os.path.join(subject_path, "tags.csv")
    
    if not os.path.exists(tags_path):
        return []
    
    tags = []
    
    with open(tags_path, 'r') as f:
        for line_num, line in enumerate(f.readlines(), 1):
            line = line.strip()
            
            if not line:
                continue
            
            try:
                tag_ts = float(line)
                tags.append(tag_ts)
                
            except ValueError:
                try:
                    tag_dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
                    tag_ts = tag_dt.timestamp()
                    tags.append(tag_ts)
                    
                except ValueError:
                    try:
                        tag_dt = datetime.fromisoformat(line.replace('T', ' '))
                        tag_ts = tag_dt.timestamp()
                        tags.append(tag_ts)
                        
                    except ValueError:
                        print(f"Línea {line_num} en tags.csv no reconocida: '{line}'")
                        continue
    
    return tags

def _parse_tag_value(raw_value):
    """Parsea un tag que puede ser numérico o fecha en texto."""
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    try:
        return float(raw_value)
    except ValueError:
        try:
            return datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            raise ValueError(f"Formato de tag no soportado: {raw_value}")
        
def segment_signal_by_tags(signal_time, signal_values, tags, window_size=60):
    """ Segmenta una señal en ventanas de tamaño fijo entre tags consecutivos."""
    
    segments = []
    
    for i in range(len(tags) - 1):
        phase_start = tags[i]
        phase_end = tags[i + 1]
        
        mask = (signal_time >= phase_start) & (signal_time < phase_end)
        phase_time = signal_time[mask]
        phase_values = signal_values[mask]
        
        if len(phase_values) == 0:
            continue
        
        phase_duration = phase_end - phase_start
        n_windows = int(phase_duration // window_size)
        
        for w in range(n_windows):
            win_start = phase_start + w * window_size
            win_end = win_start + window_size
            
            win_mask = (phase_time >= win_start) & (phase_time < win_end)
            win_values = phase_values[win_mask]
            win_time = phase_time[win_mask]
            
            if len(win_values) > 10:
                segments.append({
                    'phase_index': i,
                    'window_index': w,
                    'start_time': win_start,
                    'end_time': win_end,
                    'time': win_time,
                    'values': win_values,
                    'n_samples': len(win_values)
                })
    
    return segments

def get_phase_label(condition, phase_index, subject_id):
    """ Mapea el índice de fase a una etiqueta según el protocolo experimental. """
    
    if condition == 'STRESS':
        if subject_id.startswith('f'):
            phases = {
                0: 'rest',
                1: 'stress',
                2: 'rest',
                3: 'stress',
                4: 'stress',
                5: 'rest'
            }
        else:
            phases = {
                0: 'rest',
                1: 'stress',
                2: 'rest',
                3: 'stress',
                4: 'rest',
                5: 'stress',
                6: 'stress'
            }
        return phases.get(phase_index, 'unknown')
    
    elif condition == 'AEROBIC':
        return 'rest' if phase_index == 0 else 'aerobic'
    
    elif condition == 'ANAEROBIC':
        if phase_index == 0:
            return 'rest'
        elif phase_index % 2 == 1:
            return 'anaerobic'
        else:
            return 'rest'
    
    return 'unknown'