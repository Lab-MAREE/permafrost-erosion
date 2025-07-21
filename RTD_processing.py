# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 17:46:04 2025

@author: Hatim.BEN_SAID
"""

"""
Fonctions de traitement des données RTD pour l'analyse thermique
Analyse du dégel du pergélisol, calculs du coefficient de transfert de chaleur par convection à l'interface eau-pergélisol, gradients de température
Adapté pour l'étude de l'érosion du pergélisol côtier en laboratoire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from scipy.signal import savgol_filter
import scipy.integrate as integrate

def load_rtd_data(file_path, sheet_index=1):
    '''
    Charge les données RTD depuis un fichier Excel
    
    Intrants:
    - file_path : str - Chemin vers le fichier Excel
    - sheet_index : int - Index de la feuille à charger (défaut: 1)
    
    Extrants:
    - df : pandas DataFrame - Données avec colonnes ['Time', 'RTD1', ..., 'RTD7', 'RTD8']  (RTD8 correspond au capteur mesurant la température de l'eau dans le canal)
    
    Format des intrants:
    - Fichier Excel avec colonnes: Time, RTD1, RTD2, ..., RTD7, RTD8
    - Colonne Time en format datetime ou convertible en datetime
    '''
    excel_file = pd.ExcelFile(file_path)
    sheet_name = excel_file.sheet_names[sheet_index]
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    df['Time'] = pd.to_datetime(df['Time'])
    expected_columns = ['Time', 'RTD1', 'RTD2', 'RTD3', 'RTD4', 'RTD5', 'RTD6', 'RTD7', 'RTD8']
    
    if len(df.columns) >= len(expected_columns):
        df.columns = expected_columns[:len(df.columns)]
    
    return df

def apply_moving_average(data, window_size=50):
    '''
    Applique une moyenne mobile sur les données pour lisser le signal
    
    Intrants:
    - data : pandas Series - Série de données à lisser
    - window_size : int - Taille de la fenêtre de moyennage (défaut: 50)
    
    Extrants:
    - smoothed_data : pandas Series - Données lissées
    
    Format des intrants:
    - data doit être une Series pandas avec index numérique
    - window_size adapté selon la fréquence d'échantillonnage et le bruit
    '''
    return data.rolling(window=window_size, center=True).mean()

def preprocess_rtd_data(df, start_time, sampling_freq=50):
    '''
    Prétraite les données RTD avec découpage temporel et rééchantillonnage
    
    Intrants:
    - df : pandas DataFrame - Données RTD brutes
    - start_time : str ou pd.Timestamp - Temps de début d'analyse
    - sampling_freq : int - Fréquence d'échantillonnage cible (Hz)
    
    Extrants:
    - df_resampled : pandas DataFrame - Données prétraitées et rééchantillonnées
    - start_timestamp : pd.Timestamp - Timestamp de début converti
    
    Format des intrants:
    - df avec colonne 'Time' en datetime et colonnes RTD numériques
    - start_time: "YYYY-MM-DD HH:MM:SS" ou objet datetime
    '''
    start_timestamp = pd.to_datetime(start_time)
    df = df[df['Time'] >= start_timestamp].copy()
    df['Time'] = (df['Time'] - start_timestamp).dt.total_seconds()
    
    # Lissage des données
    rtd_columns = [col for col in df.columns if 'RTD' in col]
    for col in rtd_columns:
        df[col] = apply_moving_average(df[col])
    
    df = df.dropna()
    
    # Rééchantillonnage
    time_step = 1/sampling_freq
    new_index = pd.Index(np.arange(0, df['Time'].max() + time_step, time_step), name='Time')
    df_resampled = df.set_index('Time').reindex(new_index, method='nearest').reset_index()
    
    return df_resampled, start_timestamp

def preprocess_by_temperature_criterion(df, temp_threshold=8.0, sampling_freq=50):
    '''
    Prétraite les données en utilisant un critère de température comme déclencheur (un choix doit être faite entre un prétraitement par temps de début ou par une température seuil pour l'eau)
    
    Intrants:
    - df : pandas DataFrame - Données RTD brutes
    - temp_threshold : float - Température seuil de l'eau (°C)
    - sampling_freq : int - Fréquence d'échantillonnage cible (Hz)
    
    Extrants:
    - df_resampled : pandas DataFrame - Données prétraitées
    - start_time : pd.Timestamp - Temps de début déterminé automatiquement
    
    Format des intrants:
    - df doit contenir une colonne 'RTD_eau' avec les températures d'eau
    - temp_threshold typiquement entre 0-15°C selon les conditions expérimentales
    '''
    mask = df['RTD8'] < temp_threshold
    if not mask.any():
        raise ValueError(f"La température de l'eau n'atteint jamais {temp_threshold}°C")
    
    start_index = mask.idxmax()
    start_time = df['Time'].iloc[start_index]
    
    df = df[df['Time'] >= start_time].copy()
    df['Time'] = (df['Time'] - start_time).dt.total_seconds()
    
    # Lissage
    rtd_columns = [col for col in df.columns if 'RTD' in col]
    for col in rtd_columns:
        df[col] = apply_moving_average(df[col])
    
    df = df.dropna()
    
    # Rééchantillonnage
    time_step = 1/sampling_freq
    new_index = pd.Index(np.arange(0, df['Time'].max() + time_step, time_step), name='Time')
    df_resampled = df.set_index('Time').reindex(new_index, method='nearest').reset_index()
    
    return df_resampled, start_time

def calculate_thaw_front(df_processed, rtd_positions):
    '''
    Calcule la position du front de dégel par interpolation des températures
    
    Intrants:
    - df_processed : pandas DataFrame - Données de température prétraitées
    - rtd_positions : list - Positions des capteurs RTD (m)
    
    Extrants:
    - thaw_front : list - Position du front de dégel à chaque instant (m)
    
    Format des intrants:
    - df_processed avec colonnes 'RTD_eau', 'RTD1', ..., 'RTD7'
    - rtd_positions: [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.20] par exemple (en mètres)
    - Capteurs ordonnés spatialement de la surface vers la profondeur
    '''
    thaw_front = []
    positions = [0] + rtd_positions  # Position eau (0) + positions RTD
    
    for _, row in df_processed.iterrows():
        # Températures: eau + RTD1-7
        rtd_columns = ['RTD8'] + [f'RTD{i+1}' for i in range(len(rtd_positions))]
        temps = [row[col] for col in rtd_columns if col in row.index]
        
        if len(temps) == len(positions):
            try:
                interp_func = interp1d(positions, temps, kind='linear', fill_value='extrapolate')
                x_interp = np.linspace(positions[0], positions[-1], 1000)
                y_interp = interp_func(x_interp)
                
                # Trouver les passages à zéro
                zero_crossings = np.where(np.diff(np.sign(y_interp)))[0]
                
                if len(zero_crossings) > 0:
                    thaw_front.append(x_interp[zero_crossings[0]])
                else:
                    thaw_front.append(np.nan)
            except:
                thaw_front.append(np.nan)
        else:
            thaw_front.append(np.nan)
    
    return thaw_front

def calculate_thaw_speed(thaw_front_data, time_data, window_size=21):
    '''
    Calcule la vitesse de dégel par dérivation numérique lissée
    
    Intrants:
    - thaw_front_data : array-like - Position du front de dégel (m)
    - time_data : array-like - Vecteur temps (s)
    - window_size : int - Taille de fenêtre pour le filtre Savitzky-Golay
    
    Extrants:
    - thaw_speed : pandas Series - Vitesse de dégel (m/s)
    
    Format des intrants:
    - thaw_front_data et time_data de même longueur
    - window_size impair, adapté à la résolution temporelle
    '''
    thaw_front_series = pd.Series(thaw_front_data).ffill()
    thaw_front_smoothed = savgol_filter(thaw_front_series, window_size, 3)
    
    time_diff = pd.Series(time_data).diff()
    position_diff = pd.Series(thaw_front_smoothed).diff()
    thaw_speed = position_diff / time_diff
    
    thaw_speed = thaw_speed.replace([np.inf, -np.inf], np.nan)
    thaw_speed_smoothed = savgol_filter(thaw_speed.fillna(0), window_size, 3)
    
    return pd.Series(thaw_speed_smoothed)

def find_zero_crossing_times(df_processed, rtd_positions):
    '''
    Détermine les temps de passage à 0°C pour chaque capteur RTD
    
    Intrants:
    - df_processed : pandas DataFrame - Données de température avec colonne 'Time'
    - rtd_positions : list - Positions des capteurs RTD (m)
    
    Extrants:
    - results_df : pandas DataFrame - Temps de passage et vitesses entre capteurs
    
    Format des intrants:
    - df_processed avec colonnes 'Time', 'RTD1', ..., 'RTD7'
    - rtd_positions correspondant aux colonnes RTD
    '''
    crossing_times = []
    
    for i, pos in enumerate(rtd_positions, 1):
        rtd_col = f'RTD{i}'
        if rtd_col in df_processed.columns:
            zero_crossings = np.where(np.diff(np.signbit(df_processed[rtd_col])))[0]
            
            if len(zero_crossings) > 0:
                crossing_time = df_processed['Time'].iloc[zero_crossings[0]]
                crossing_times.append({
                    'RTD': rtd_col,
                    'Position_m': pos,
                    'Temps_passage_s': crossing_time
                })
    
    results_df = pd.DataFrame(crossing_times)
    
    # Calcul des vitesses entre RTDs adjacents
    if len(results_df) > 1:
        results_df['Vitesse_m_s'] = np.nan
        for i in range(len(results_df)-1):
            delta_pos = results_df['Position_m'].iloc[i+1] - results_df['Position_m'].iloc[i]
            delta_time = results_df['Temps_passage_s'].iloc[i+1] - results_df['Temps_passage_s'].iloc[i]
            if delta_time > 0:
                velocity = delta_pos / delta_time
                results_df.loc[i, 'Vitesse_m_s'] = velocity
    
    return results_df

def calculate_spatial_temperature_gradients(df_processed, rtd_positions):
    '''
    Calcule les gradients de température spatiaux entre capteurs adjacents
    
    Intrants:
    - df_processed : pandas DataFrame - Données de température prétraitées
    - rtd_positions : list - Positions des capteurs RTD (m)
    
    Extrants:
    - gradient_stats : pandas DataFrame - Statistiques des gradients (min, max, moyenne, std)
    - gradient_data : pandas DataFrame - Séries temporelles des gradients
    
    Format des intrants:
    - df_processed avec colonnes 'RTD_eau', 'RTD1', ..., 'RTD7'
    - rtd_positions ordonnées de la surface vers la profondeur
    '''
    positions = [0] + rtd_positions  # Position eau + RTD
    gradient_columns = []
    
    for i in range(len(positions)-1):
        if i == 0:
            col_name = f"Gradient_eau_RTD1"
            df_processed[col_name] = (df_processed['RTD8'] - df_processed['RTD1']) / (positions[i+1] - positions[i])
        else:
            col_name = f"Gradient_RTD{i}_RTD{i+1}"
            df_processed[col_name] = (df_processed[f'RTD{i}'] - df_processed[f'RTD{i+1}']) / (positions[i+1] - positions[i])
        
        gradient_columns.append(col_name)
    
    # Statistiques des gradients
    gradient_stats = pd.DataFrame()
    for col in gradient_columns:
        gradient_stats[col] = [
            df_processed[col].min(),
            df_processed[col].max(),
            df_processed[col].mean(),
            df_processed[col].std()
        ]
    gradient_stats.index = ['Min_C_per_m', 'Max_C_per_m', 'Mean_C_per_m', 'Std_C_per_m']
    
    return gradient_stats, df_processed[gradient_columns]

def calculate_temporal_temperature_gradients(df_processed):
    '''
    Calcule les gradients temporels de température (dT/dt) pour chaque capteur
    
    Intrants:
    - df_processed : pandas DataFrame - Données de température avec colonne 'Time'
    
    Extrants:
    - temporal_gradient_stats : pandas DataFrame - Statistiques des gradients temporels
    - temporal_gradient_data : pandas DataFrame - Séries temporelles des gradients temporels
    
    Format des intrants:
    - df_processed avec colonne 'Time' et colonnes de température RTD
    - Échantillonnage temporel régulier recommandé
    '''
    temp_columns = ['RTD8'] + [f'RTD{i}' for i in range(1, 8)]
    temp_columns = [col for col in temp_columns if col in df_processed.columns]
    
    dt_mean = np.mean(np.diff(df_processed['Time']))
    gradient_columns = []
    
    for col in temp_columns:
        grad_col_name = f"dT_dt_{col}"
        gradient_columns.append(grad_col_name)
        df_processed[grad_col_name] = df_processed[col].diff() / dt_mean
    
    # Statistiques des gradients temporels
    temporal_gradient_stats = pd.DataFrame()
    for col in gradient_columns:
        valid_data = df_processed[col].replace([np.inf, -np.inf], np.nan).dropna()
        temporal_gradient_stats[col] = [
            valid_data.min(),
            valid_data.max(),
            valid_data.mean(),
            valid_data.std()
        ]
    
    temporal_gradient_stats.index = ['Min_C_per_s', 'Max_C_per_s', 'Mean_C_per_s', 'Std_C_per_s']
    
    return temporal_gradient_stats, df_processed[gradient_columns]

def calculate_heat_transfer(temperatures, positions, time, thaw_depth, 
                          k_frozen=2.67, c_frozen=1600000, latent_heat=63.17e6):
    '''
    Calcule le transfert de chaleur avec composantes sensible et latente
    
    Intrants:
    - temperatures : numpy array (n_sensors x n_time) - Matrice des températures (°C)
    - positions : list - Positions des capteurs (m)
    - time : numpy array - Vecteur temps (s)
    - thaw_depth : numpy array - Profondeur du front de dégel (m)
    - k_frozen : float - Conductivité thermique du sol gelé (W/m·K)
    - c_frozen : float - Capacité calorifique volumique du sol gelé (J/m³·K)
    - latent_heat : float - Chaleur latente volumique de fusion (J/m³)
    
    Extrants:
    - Q_total : numpy array - Flux de chaleur total (W/m²)
    - Q_sensible : numpy array - Flux de chaleur sensible (W/m²)
    - Q_latent : numpy array - Flux de chaleur latente (W/m²)
    - h_coefficient : numpy array - Coefficient de transfert convectif (W/m²·K)
    - h_stats : dict - Statistiques du coefficient h
    
    Format des intrants:
    - temperatures[0] = température de l'eau de surface
    - positions[0] = 0 (surface), positions[1:] = profondeurs des RTD
    - Tous les arrays de même longueur temporelle
    '''
    n_time = len(time)
    n_spaces = len(positions) - 2
    
    Q_sensible = np.zeros(n_time)
    Q_latent = np.zeros(n_time)
    h_coefficient = np.zeros(n_time)
    
    # Calcul des flux par espaces inter-capteurs
    heat_flux_matrix = np.zeros((n_time, n_spaces))
    
    for t_idx in range(1, n_time):
        dt = time[t_idx] - time[t_idx - 1]
        
        # Flux de chaleur sensible entre capteurs
        for i in range(n_spaces):
            if i == 0:  # Entre surface et premier RTD
                delta_T = temperatures[0, t_idx] - temperatures[i+1, t_idx]
                delta_x = abs(thaw_depth[t_idx] - positions[i+1]) if not np.isnan(thaw_depth[t_idx]) else abs(positions[i+1] - positions[i])
            else:  # Entre RTDs adjacents
                delta_T = temperatures[i, t_idx] - temperatures[i+1, t_idx]
                delta_x = abs(positions[i+1] - positions[i])
            
            if delta_x > 0:
                heat_flux_matrix[t_idx, i] = k_frozen * delta_T / delta_x
        
        Q_sensible[t_idx] = np.sum(heat_flux_matrix[t_idx, :])
        
        # Flux de chaleur latente (changement de phase)
        if t_idx > 0 and not np.isnan(thaw_depth[t_idx]) and not np.isnan(thaw_depth[t_idx-1]):
            dthaw_dt = (thaw_depth[t_idx] - thaw_depth[t_idx-1]) / dt
            Q_latent[t_idx] = latent_heat * max(0, dthaw_dt)
        
        # Flux total et coefficient de transfert
        Q_total_t = Q_sensible[t_idx] + Q_latent[t_idx]
        TD = temperatures[0, t_idx] - 0  # Différence avec température de fusion
        
        if abs(TD) > 0.1:  # Éviter division par zéro
            h_coefficient[t_idx] = Q_total_t / TD
    
    Q_total = Q_sensible + Q_latent
    
    # Statistiques du coefficient h
    h_valid = h_coefficient[h_coefficient > 0]
    h_stats = {
        'mean': np.mean(h_valid) if len(h_valid) > 0 else 0,
        'median': np.median(h_valid) if len(h_valid) > 0 else 0,
        'std': np.std(h_valid) if len(h_valid) > 0 else 0,
        'min': np.min(h_valid) if len(h_valid) > 0 else 0,
        'max': np.max(h_valid) if len(h_valid) > 0 else 0
    }
    
    return Q_total, Q_sensible, Q_latent, h_coefficient, h_stats

def calculate_cumulative_heat(Q_sensible, Q_latent, time):
    '''
    Calcule les chaleurs cumulatives transférées
    
    Intrants:
    - Q_sensible : numpy array - Flux de chaleur sensible (W/m²)
    - Q_latent : numpy array - Flux de chaleur latente (W/m²)
    - time : numpy array - Vecteur temps (s)
    
    Extrants:
    - cumulative_sensible : float - Chaleur sensible cumulative (J/m²)
    - cumulative_latent : float - Chaleur latente cumulative (J/m²)
    - cumulative_total : float - Chaleur totale cumulative (J/m²)
    
    Format des intrants:
    - Arrays de même longueur
    - time avec pas de temps régulier ou variable
    '''
    dt_mean = np.mean(np.diff(time))
    
    cumulative_sensible = np.sum(Q_sensible) * dt_mean
    cumulative_latent = np.sum(Q_latent) * dt_mean
    cumulative_total = cumulative_sensible + cumulative_latent
    
    return cumulative_sensible, cumulative_latent, cumulative_total

def analyze_thermal_properties(df_processed, rtd_positions, thermal_params=None):
    '''
    Analyse complète des propriétés thermiques du système
    
    Intrants:
    - df_processed : pandas DataFrame - Données de température prétraitées
    - rtd_positions : list - Positions des capteurs RTD (m)
    - thermal_params : dict - Paramètres thermiques optionnels
        - k_frozen : conductivité sol gelé (W/m·K)
        - c_frozen : capacité calorifique sol gelé (J/m³·K)
        - latent_heat : chaleur latente de fusion (J/m³)
    
    Extrants:
    - results : dict - Dictionnaire complet des résultats d'analyse thermique
    
    Format des intrants:
    - df_processed avec colonnes 'Time', 'RTD_eau', 'RTD1', ..., et front de dégel calculé
    - thermal_params optionnel, sinon valeurs par défaut pour pergélisol
    '''
    # Paramètres thermiques par défaut pour pergélisol
    if thermal_params is None:
        thermal_params = {
            'k_frozen': 2.67,      # W/(m·K)
            'c_frozen': 1600000,   # J/(m³·K)
            'latent_heat': 63.17e6 # J/m³
        }
    
    # Préparation des données
    temp_columns = ['RTD8'] + [f'RTD{i}' for i in range(1, 8)]
    temp_columns = [col for col in temp_columns if col in df_processed.columns]
    
    temperatures = df_processed[temp_columns].values.T
    time = df_processed['Time'].values
    positions = [0] + rtd_positions
    
    # Calcul du front de dégel si absent
    if 'thaw_front' not in df_processed.columns:
        thaw_front = calculate_thaw_front(df_processed, rtd_positions)
        df_processed['thaw_front'] = thaw_front
    
    thaw_depth = df_processed['thaw_front'].values
    
    # Analyses des gradients
    spatial_grad_stats, spatial_grad_data = calculate_spatial_temperature_gradients(df_processed, rtd_positions)
    temporal_grad_stats, temporal_grad_data = calculate_temporal_temperature_gradients(df_processed)
    
    # Analyse du transfert de chaleur
    Q_total, Q_sensible, Q_latent, h_coeff, h_stats = calculate_heat_transfer(
        temperatures, positions, time, thaw_depth, 
        thermal_params['k_frozen'], thermal_params['c_frozen'], thermal_params['latent_heat']
    )
    
    # Chaleurs cumulatives
    cum_sensible, cum_latent, cum_total = calculate_cumulative_heat(Q_sensible, Q_latent, time)
    
    # Vitesse de dégel
    thaw_speed = calculate_thaw_speed(thaw_depth, time)
    
    # Temps de passage à 0°C
    crossing_times = find_zero_crossing_times(df_processed, rtd_positions)
    
    # Compilation des résultats
    results = {
        'processed_data': df_processed,
        'thermal_parameters': thermal_params,
        'temperatures_matrix': temperatures,
        'time_vector': time,
        'positions': positions,
        'thaw_front': thaw_depth,
        'thaw_speed': thaw_speed,
        'zero_crossing_times': crossing_times,
        'spatial_gradients': {
            'statistics': spatial_grad_stats,
            'data': spatial_grad_data
        },
        'temporal_gradients': {
            'statistics': temporal_grad_stats,
            'data': temporal_grad_data
        },
        'heat_transfer': {
            'Q_total': Q_total,
            'Q_sensible': Q_sensible,
            'Q_latent': Q_latent,
            'h_coefficient': h_coeff,
            'h_statistics': h_stats
        },
        'cumulative_heat': {
            'sensible': cum_sensible,
            'latent': cum_latent,
            'total': cum_total
        }
    }
    
    return results

def save_thermal_analysis_results(results, output_path, test_id):
    '''
    Sauvegarde les résultats d'analyse thermique dans un fichier Excel multi-feuilles
    
    Intrants:
    - results : dict - Dictionnaire des résultats depuis analyze_thermal_properties()
    - output_path : str - Chemin de sauvegarde du fichier Excel
    - test_id : str - Identifiant du test pour nommer le fichier
    
    Extrants:
    - excel_path : str - Chemin complet du fichier Excel créé
    
    Format des intrants:
    - results doit contenir les clés standard de analyze_thermal_properties()
    - output_path: répertoire existant pour la sauvegarde
    '''
    from datetime import datetime
    import os
    
    # Nom du fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"{test_id}_analyse_thermique_{timestamp}.xlsx"
    excel_path = os.path.join(output_path, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Données principales
        if 'processed_data' in results:
            results['processed_data'].to_excel(writer, sheet_name='Donnees_traitees', index=False)
        
        # Paramètres thermiques
        if 'thermal_parameters' in results:
            pd.DataFrame.from_dict(results['thermal_parameters'], orient='index', 
                                 columns=['Valeur']).to_excel(writer, sheet_name='Parametres_thermiques')
        
        # Temps de passage
        if 'zero_crossing_times' in results:
            results['zero_crossing_times'].to_excel(writer, sheet_name='Temps_passage_0C', index=False)
        
        # Gradients spatiaux
        if 'spatial_gradients' in results:
            results['spatial_gradients']['statistics'].to_excel(writer, sheet_name='Gradients_spatiaux_stats')
        
        # Gradients temporels
        if 'temporal_gradients' in results:
            results['temporal_gradients']['statistics'].to_excel(writer, sheet_name='Gradients_temporels_stats')
        
        # Statistiques transfert de chaleur
        if 'heat_transfer' in results and 'h_statistics' in results['heat_transfer']:
            pd.DataFrame.from_dict(results['heat_transfer']['h_statistics'], orient='index',
                                 columns=['Valeur']).to_excel(writer, sheet_name='Statistiques_h')
        
        # Chaleurs cumulatives
        if 'cumulative_heat' in results:
            pd.DataFrame.from_dict(results['cumulative_heat'], orient='index',
                                 columns=['Valeur_J_per_m2']).to_excel(writer, sheet_name='Chaleurs_cumulatives')
    
    return excel_path