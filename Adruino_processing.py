# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 18:15:26 2025

@author: Olorunfemi_Adeyemi.OMONIGBEHIN
"""

"""
Fonctions de traitement des données ADV (Acoustic Doppler Velocimeter)
Analyse des vitesses d'écoulement, filtrage qualité, statistiques hydrodynamiques
Adapté pour l'étude de l'érosion du pergélisol côtier en laboratoire
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import signal
from scipy.fft import rfft, rfftfreq
import scipy.integrate as integrate

def load_adv_data(dat_file, hdr_file):
    '''
    Charge les données ADV depuis les fichiers .dat et .hdr
    
    Intrants:
    - dat_file : str - Chemin vers le fichier de données .dat
    - hdr_file : str - Chemin vers le fichier d'en-tête .hdr
    
    Extrants:
    - df : pandas DataFrame - Données ADV avec colonnes:
        ['Timestamp', 'U_velocity', 'V_velocity', 'W_velocity', 'SNR', 'Correlation']
    
    Format des intrants:
    - Fichier .dat: colonnes séparées par espaces, temps en secondes en col 0,
      vitesses U,V,W en col 3,4,5, SNR en col 11-14, corrélation en col 15-18
    - Fichier .hdr: ligne 5 contient le temps de début au format "YYYY-MM-DD H:MM:SS AM/PM"
    '''
    # Lecture du fichier header pour extraire le temps de début
    with open(hdr_file, 'r') as file:
        hdr_lines = file.readlines()
    
    try:
        start_time_str = hdr_lines[4].split("Time of first measurement")[1].strip()
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %I:%M:%S %p")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Impossible d'extraire le temps de début du fichier .hdr: {e}")
    
    # Lecture du fichier de données
    with open(dat_file, 'r') as file:
        data_lines = file.readlines()
    
    data = []
    for line in data_lines:
        values = line.split()
        
        if len(values) >= 19:
            time_seconds = float(values[0])
            u_velocity = float(values[3])  # Vitesse X
            v_velocity = float(values[4])  # Vitesse Y  
            w_velocity = float(values[5])  # Vitesse Z
            snr_beams = [float(values[11]), float(values[12]), float(values[13]), float(values[14])]
            correlation_beams = [float(values[15]), float(values[16]), float(values[17]), float(values[18])]
            
            timestamp = start_time + timedelta(seconds=time_seconds)
            data.append([timestamp, u_velocity, v_velocity, w_velocity, snr_beams, correlation_beams])
    
    df = pd.DataFrame(data, columns=['Timestamp', 'U_velocity', 'V_velocity', 'W_velocity', 'SNR', 'Correlation'])
    
    return df

def preprocess_adv_data(df, start_time, duration_minutes):
    '''
    Prétraite les données ADV avec découpage temporel
    
    Intrants:
    - df : pandas DataFrame - Données ADV depuis load_adv_data()
    - start_time : str ou datetime - Temps de début d'analyse
    - duration_minutes : float - Durée d'analyse en minutes
    
    Extrants:
    - trimmed_df : pandas DataFrame - Données découpées temporellement
    - start_timestamp : datetime - Timestamp de début converti
    - end_timestamp : datetime - Timestamp de fin calculé
    
    Format des intrants:
    - start_time: "YYYY-MM-DD HH:MM:SS.ffffff" ou objet datetime
    - duration_minutes > 0
    '''
    if isinstance(start_time, str):
        start_timestamp = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
    else:
        start_timestamp = start_time
    
    end_timestamp = start_timestamp + timedelta(minutes=duration_minutes)
    
    trimmed_df = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)].copy()
    
    return trimmed_df, start_timestamp, end_timestamp

def filter_adv_data_quality(df, snr_threshold=9, correlation_threshold=95):
    '''
    Filtre les données ADV selon les critères de qualité SNR et corrélation
    
    Intrants:
    - df : pandas DataFrame - Données ADV avec colonnes SNR et Correlation
    - snr_threshold : float - Seuil minimum SNR pour tous les faisceaux (défaut: 9)
    - correlation_threshold : float - Seuil minimum corrélation en % (défaut: 95)
    
    Extrants:
    - filtered_df : pandas DataFrame - Données filtrées selon critères qualité
    
    Format des intrants:
    - df['SNR'] contient des listes de 4 valeurs (un par faisceau)
    - df['Correlation'] contient des listes de 4 valeurs en pourcentage
    - Seuils typiques: SNR ≥ 9-15, Corrélation ≥ 90-95%
    '''
    filtered_df = df[
        df['SNR'].apply(lambda x: all(snr >= snr_threshold for snr in x)) &
        df['Correlation'].apply(lambda x: all(corr >= correlation_threshold for corr in x))
    ].copy()
    
    return filtered_df

def calculate_velocity_statistics(df):
    '''
    Calcule les statistiques des composantes de vitesse
    
    Intrants:
    - df : pandas DataFrame - Données ADV filtrées
    
    Extrants:
    - stats : dict - Statistiques pour chaque composante de vitesse
        - mean, std, min, max, rms pour U, V, W
        - velocity_magnitude_mean : magnitude moyenne du vecteur vitesse
        - turbulence_intensity : intensité de turbulence par composante
    
    Format des intrants:
    - df avec colonnes 'U_velocity', 'V_velocity', 'W_velocity' en m/s
    '''
    velocity_components = ['U_velocity', 'V_velocity', 'W_velocity']
    stats = {}
    
    for component in velocity_components:
        comp_short = component[0]  # U, V, ou W
        values = df[component].values
        
        stats[f'{comp_short}_mean'] = np.mean(values)
        stats[f'{comp_short}_std'] = np.std(values)
        stats[f'{comp_short}_min'] = np.min(values)
        stats[f'{comp_short}_max'] = np.max(values)
        stats[f'{comp_short}_rms'] = np.sqrt(np.mean(values**2))
        
        # Intensité de turbulence (écart-type / vitesse moyenne)
        if abs(stats[f'{comp_short}_mean']) > 1e-6:
            stats[f'{comp_short}_turbulence_intensity'] = stats[f'{comp_short}_std'] / abs(stats[f'{comp_short}_mean'])
        else:
            stats[f'{comp_short}_turbulence_intensity'] = np.nan
    
    # Magnitude du vecteur vitesse
    velocity_magnitude = np.sqrt(df['U_velocity']**2 + df['V_velocity']**2 + df['W_velocity']**2)
    stats['velocity_magnitude_mean'] = np.mean(velocity_magnitude)
    stats['velocity_magnitude_std'] = np.std(velocity_magnitude)
    stats['velocity_magnitude_max'] = np.max(velocity_magnitude)
    
    return stats

def calculate_reynolds_stresses(df):
    '''
    Calcule les contraintes de Reynolds et paramètres de turbulence
    
    Intrants:
    - df : pandas DataFrame - Données ADV avec composantes de vitesse
    
    Extrants:
    - reynolds_stats : dict - Statistiques de turbulence
        - u_prime_v_prime : contrainte de cisaillement turbulente
        - tke : énergie cinétique turbulente
        - fluctuation_velocities : vitesses de fluctuation RMS
    
    Format des intrants:
    - df avec colonnes 'U_velocity', 'V_velocity', 'W_velocity' 
    - Données à fréquence constante pour analyse turbulente
    '''
    # Vitesses moyennes
    U_mean = df['U_velocity'].mean()
    V_mean = df['V_velocity'].mean()
    W_mean = df['W_velocity'].mean()
    
    # Fluctuations de vitesse
    u_prime = df['U_velocity'] - U_mean
    v_prime = df['V_velocity'] - V_mean
    w_prime = df['W_velocity'] - W_mean
    
    # Contraintes de Reynolds
    reynolds_stats = {
        'u_prime_squared': np.mean(u_prime**2),
        'v_prime_squared': np.mean(v_prime**2),
        'w_prime_squared': np.mean(w_prime**2),
        'u_prime_v_prime': np.mean(u_prime * v_prime),  # Contrainte de cisaillement principale
        'u_prime_w_prime': np.mean(u_prime * w_prime),
        'v_prime_w_prime': np.mean(v_prime * w_prime)
    }
    
    # Énergie cinétique turbulente (TKE)
    reynolds_stats['tke'] = 0.5 * (reynolds_stats['u_prime_squared'] + 
                                  reynolds_stats['v_prime_squared'] + 
                                  reynolds_stats['w_prime_squared'])
    
    # Vitesses de fluctuation RMS
    reynolds_stats['u_rms'] = np.sqrt(reynolds_stats['u_prime_squared'])
    reynolds_stats['v_rms'] = np.sqrt(reynolds_stats['v_prime_squared'])
    reynolds_stats['w_rms'] = np.sqrt(reynolds_stats['w_prime_squared'])
    
    return reynolds_stats

def calculate_velocity_spectra(df, sampling_frequency):
    '''
    Calcule les spectres de puissance des composantes de vitesse
    
    Intrants:
    - df : pandas DataFrame - Données ADV avec échantillonnage régulier
    - sampling_frequency : float - Fréquence d'échantillonnage (Hz)
    
    Extrants:
    - spectra_results : dict - Spectres pour chaque composante
        - frequencies : vecteur fréquence commun
        - U_spectrum, V_spectrum, W_spectrum : densités spectrales (m²/s²)/Hz
    
    Format des intrants:
    - df avec données à fréquence constante
    - sampling_frequency typiquement 25-100 Hz pour ADV
    '''
    velocity_components = ['U_velocity', 'V_velocity', 'W_velocity']
    spectra_results = {}
    
    # Calcul du spectre pour la première composante pour obtenir les fréquences
    u_data = df['U_velocity'].values - df['U_velocity'].mean()
    f, psd_u = signal.welch(u_data, fs=sampling_frequency, window='hann', nperseg=1024)
    
    spectra_results['frequencies'] = f
    spectra_results['U_spectrum'] = psd_u
    
    # Calcul pour les autres composantes
    for component in velocity_components[1:]:  # V et W
        comp_data = df[component].values - df[component].mean()
        _, psd = signal.welch(comp_data, fs=sampling_frequency, window='hann', nperseg=1024)
        comp_short = component[0]
        spectra_results[f'{comp_short}_spectrum'] = psd
    
    return spectra_results

def detect_velocity_peaks(df, component='U_velocity', prominence_factor=2.0):
    '''
    Détecte les pics dans les séries temporelles de vitesse
    
    Intrants:
    - df : pandas DataFrame - Données ADV
    - component : str - Composante à analyser ('U_velocity', 'V_velocity', 'W_velocity')
    - prominence_factor : float - Facteur de proéminence relative à l'écart-type
    
    Extrants:
    - peaks_info : dict - Informations sur les pics détectés
        - peak_indices : indices des pics
        - peak_values : valeurs aux pics
        - peak_times : timestamps des pics
        - num_peaks : nombre total de pics
    
    Format des intrants:
    - component doit être une colonne valide de df
    - prominence_factor > 0, typiquement 1.5-3.0
    '''
    velocity_data = df[component].values
    std_velocity = np.std(velocity_data)
    prominence = prominence_factor * std_velocity
    
    # Détection des pics
    peaks, properties = signal.find_peaks(velocity_data, prominence=prominence)
    
    peaks_info = {
        'peak_indices': peaks,
        'peak_values': velocity_data[peaks],
        'peak_times': df['Timestamp'].iloc[peaks].values,
        'num_peaks': len(peaks),
        'prominence_used': prominence,
        'component_analyzed': component
    }
    
    return peaks_info

def calculate_flow_direction(df):
    '''
    Calcule la direction principale de l'écoulement
    
    Intrants:
    - df : pandas DataFrame - Données ADV avec composantes U, V
    
    Extrants:
    - flow_direction : dict - Paramètres directionnels de l'écoulement
        - mean_direction_deg : direction moyenne en degrés (0° = +X, 90° = +Y)
        - direction_std_deg : écart-type de la direction
        - resultant_velocity : vitesse résultante moyenne
        - directional_steadiness : coefficient de stabilité directionnelle
    
    Format des intrants:
    - df avec colonnes 'U_velocity' et 'V_velocity' (composantes horizontales)
    '''
    U = df['U_velocity'].values
    V = df['V_velocity'].values
    
    # Direction instantanée en radians puis degrés
    directions_rad = np.arctan2(V, U)
    directions_deg = np.degrees(directions_rad)
    
    # Direction moyenne vectorielle
    mean_U = np.mean(U)
    mean_V = np.mean(V)
    mean_direction_rad = np.arctan2(mean_V, mean_U)
    mean_direction_deg = np.degrees(mean_direction_rad)
    
    # Vitesse résultante
    resultant_velocity = np.sqrt(mean_U**2 + mean_V**2)
    
    # Magnitude de vitesse horizontale instantanée
    horizontal_velocity = np.sqrt(U**2 + V**2)
    mean_horizontal_velocity = np.mean(horizontal_velocity)
    
    # Coefficient de stabilité directionnelle
    directional_steadiness = resultant_velocity / mean_horizontal_velocity if mean_horizontal_velocity > 0 else 0
    
    # Écart-type circulaire de la direction
    direction_std_deg = np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * directions_rad))))))
    
    flow_direction = {
        'mean_direction_deg': mean_direction_deg,
        'direction_std_deg': direction_std_deg,
        'resultant_velocity': resultant_velocity,
        'mean_horizontal_velocity': mean_horizontal_velocity,
        'directional_steadiness': directional_steadiness
    }
    
    return flow_direction

def calculate_shear_velocity(reynolds_stress, density=1000):
    '''
    Calcule la vitesse de cisaillement à partir des contraintes de Reynolds
    
    Intrants:
    - reynolds_stress : float - Contrainte de Reynolds u'v' (m²/s²)
    - density : float - Masse volumique du fluide (kg/m³) (défaut: 1000 pour l'eau)
    
    Extrants:
    - u_star : float - Vitesse de cisaillement (m/s)
    - shear_stress : float - Contrainte de cisaillement (Pa)
    
    Format des intrants:
    - reynolds_stress depuis calculate_reynolds_stresses()
    - density > 0
    '''
    # Contrainte de cisaillement
    shear_stress = density * abs(reynolds_stress)
    
    # Vitesse de cisaillement
    u_star = np.sqrt(abs(reynolds_stress))
    
    return u_star, shear_stress

def analyze_velocity_data_complete(df, sampling_frequency, density=1000):
    '''
    Analyse complète des données de vitesse ADV
    
    Intrants:
    - df : pandas DataFrame - Données ADV filtrées
    - sampling_frequency : float - Fréquence d'échantillonnage (Hz)
    - density : float - Masse volumique du fluide (kg/m³)
    
    Extrants:
    - analysis_results : dict - Résultats complets d'analyse
        - basic_statistics : statistiques de base des vitesses
        - reynolds_statistics : contraintes de Reynolds et turbulence
        - spectral_analysis : spectres de puissance
        - flow_direction : analyse directionnelle
        - shear_parameters : paramètres de cisaillement
    
    Format des intrants:
    - df depuis filter_adv_data_quality()
    - sampling_frequency selon configuration ADV
    '''
    # Statistiques de base
    basic_stats = calculate_velocity_statistics(df)
    
    # Analyse de turbulence
    reynolds_stats = calculate_reynolds_stresses(df)
    
    # Analyse spectrale
    spectra_results = calculate_velocity_spectra(df, sampling_frequency)
    
    # Direction d'écoulement
    flow_dir = calculate_flow_direction(df)
    
    # Paramètres de cisaillement
    u_star, shear_stress = calculate_shear_velocity(reynolds_stats['u_prime_v_prime'], density)
    
    # Détection de pics pour chaque composante
    peaks_u = detect_velocity_peaks(df, 'U_velocity')
    peaks_v = detect_velocity_peaks(df, 'V_velocity')
    peaks_w = detect_velocity_peaks(df, 'W_velocity')
    
    analysis_results = {
        'basic_statistics': basic_stats,
        'reynolds_statistics': reynolds_stats,
        'spectral_analysis': spectra_results,
        'flow_direction': flow_dir,
        'shear_parameters': {
            'u_star': u_star,
            'shear_stress': shear_stress,
            'reynolds_stress_uv': reynolds_stats['u_prime_v_prime']
        },
        'peak_detection': {
            'U_component': peaks_u,
            'V_component': peaks_v,
            'W_component': peaks_w
        },
        'data_quality': {
            'num_valid_points': len(df),
            'sampling_frequency': sampling_frequency,
            'measurement_duration': (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds(),
            'data_density': density
        }
    }
    
    return analysis_results

def export_adv_results_to_dataframe(analysis_results):
    '''
    Exporte les résultats d'analyse ADV vers des DataFrames pour sauvegarde
    
    Intrants:
    - analysis_results : dict - Résultats depuis analyze_velocity_data_complete()
    
    Extrants:
    - summary_df : pandas DataFrame - Résumé des paramètres principaux
    - detailed_df : pandas DataFrame - Tous les paramètres calculés
    
    Format des intrants:
    - analysis_results avec structure standard de analyze_velocity_data_complete()
    '''
    # DataFrame résumé avec paramètres clés
    summary_data = {
        'Paramètre': [
            'Vitesse U moyenne (m/s)',
            'Vitesse V moyenne (m/s)', 
            'Vitesse W moyenne (m/s)',
            'Magnitude vitesse moyenne (m/s)',
            'Énergie cinétique turbulente (m²/s²)',
            'Contrainte de Reynolds u\'v\' (m²/s²)',
            'Vitesse de cisaillement u* (m/s)',
            'Direction moyenne écoulement (°)',
            'Stabilité directionnelle (-)',
            'Intensité turbulence U (%)',
            'Nombre de points valides'
        ],
        'Valeur': [
            analysis_results['basic_statistics']['U_mean'],
            analysis_results['basic_statistics']['V_mean'],
            analysis_results['basic_statistics']['W_mean'],
            analysis_results['basic_statistics']['velocity_magnitude_mean'],
            analysis_results['reynolds_statistics']['tke'],
            analysis_results['reynolds_statistics']['u_prime_v_prime'],
            analysis_results['shear_parameters']['u_star'],
            analysis_results['flow_direction']['mean_direction_deg'],
            analysis_results['flow_direction']['directional_steadiness'],
            analysis_results['basic_statistics']['U_turbulence_intensity'] * 100,
            analysis_results['data_quality']['num_valid_points']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # DataFrame détaillé avec tous les paramètres
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, np.number)):
                items.append((new_key, v))
        return dict(items)
    
    flat_results = flatten_dict(analysis_results)
    detailed_data = [{'Paramètre': key, 'Valeur': value} for key, value in flat_results.items()]
    detailed_df = pd.DataFrame(detailed_data)
    
    return summary_df, detailed_df

def create_time_relative_dataframe(df, start_timestamp):
    '''
    Crée un DataFrame avec temps relatif en secondes pour export
    
    Intrants:
    - df : pandas DataFrame - Données ADV avec timestamps absolus
    - start_timestamp : datetime - Temps de référence (t=0)
    
    Extrants:
    - df_relative : pandas DataFrame - Données avec colonne 'Time_s' relative
    
    Format des intrants:
    - df avec colonne 'Timestamp' 
    - start_timestamp doit être datetime
    '''
    df_relative = df.copy()
    df_relative['Time_s'] = (df_relative['Timestamp'] - start_timestamp).dt.total_seconds()
    
    # Réorganiser les colonnes pour mettre Time_s en premier
    cols = ['Time_s'] + [col for col in df_relative.columns if col != 'Time_s']
    df_relative = df_relative[cols]
    
    return df_relative