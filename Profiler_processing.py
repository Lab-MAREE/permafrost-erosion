# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 18:20:23 2025

@author: Olorunfemi_Adeyemi.OMONIGBEHIN
"""

"""
Fonctions de traitement des données de profileur acoustique Doppler (Vectrino Profiler)
Analyse des profils de vitesse, filtrage qualité, statistiques par cellules
Adapté pour l'étude de l'érosion du pergélisol côtier en laboratoire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime, timedelta
from scipy import signal
from scipy.interpolate import interp1d
import scipy.integrate as integrate

def matlab_datenum_to_datetime(matlab_datenum):
    '''
    Convertit les timestamps Matlab (datenum) vers datetime Python
    
    Intrants:
    - matlab_datenum : float ou array - Timestamp(s) Matlab
    
    Extrants:
    - python_datetime : datetime ou list - Timestamp(s) Python
    
    Format des intrants:
    - matlab_datenum format standard Matlab (jours depuis 1er janvier 0000)
    '''
    if isinstance(matlab_datenum, (list, np.ndarray)):
        return [datetime.fromordinal(int(dn)) + timedelta(days=dn % 1) - timedelta(days=366) 
                for dn in matlab_datenum]
    else:
        return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)

def load_profiler_data(mat_file):
    '''
    Charge les données du profileur depuis un fichier .mat
    
    Intrants:
    - mat_file : str - Chemin vers le fichier .mat du Vectrino Profiler
    
    Extrants:
    - data_dict : dict - Dictionnaire des données chargées
        - timestamps : array de datetime
        - velocity_profiles : dict avec VelX, VelY, VelZ
        - range_data : array des profondeurs/distances
        - snr_data : dict avec données SNR par faisceau
        - correlation_data : dict avec données corrélation par faisceau
        - config : dict avec configuration instrument
    
    Format des intrants:
    - Fichier .mat standard Vectrino Profiler avec sections Data et Config
    '''
    print(f'Chargement des données depuis: {mat_file}')
    mat_data = loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    
    data_dict = {
        'timestamps': [],
        'velocity_profiles': {},
        'range_data': [],
        'snr_data': {},
        'correlation_data': {},
        'config': {}
    }
    
    # Extraction de la configuration
    if 'Config\x00\x00' in mat_data:
        data_dict['config'] = mat_data['Config\x00\x00']
    
    # Extraction des données principales
    if 'Data\x00\x00\x00\x00' in mat_data:
        data_section = mat_data['Data\x00\x00\x00\x00']
        
        # Extraction des timestamps
        if hasattr(data_section, 'Profiles_HostTimeMatlab'):
            host_time_matlab = data_section.Profiles_HostTimeMatlab
            data_dict['timestamps'] = matlab_datenum_to_datetime(host_time_matlab)
        
        # Extraction des profils de vitesse
        velocity_components = ['VelX', 'VelY', 'VelZ1']
        for vel_comp in velocity_components:
            field_name = f'Profiles_{vel_comp}'
            if field_name in data_section._fieldnames:
                content = getattr(data_section, field_name)
                if isinstance(content, np.ndarray) and content.ndim == 2:
                    data_dict['velocity_profiles'][vel_comp] = content
        
        # Extraction des données de distance/profondeur
        if 'Profiles_Range' in data_section._fieldnames:
            data_dict['range_data'] = getattr(data_section, 'Profiles_Range')
        
        # Extraction des données SNR et corrélation
        for beam in range(1, 4):  # Faisceaux 1, 2, 3
            snr_field = f'Profiles_SNRBeam{beam}'
            cor_field = f'Profiles_CorBeam{beam}'
            
            if snr_field in data_section._fieldnames:
                data_dict['snr_data'][f'Beam{beam}'] = getattr(data_section, snr_field)
            
            if cor_field in data_section._fieldnames:
                data_dict['correlation_data'][f'Beam{beam}'] = getattr(data_section, cor_field)
    
    return data_dict

def preprocess_profiler_data(data_dict, start_time, duration_minutes):
    '''
    Prétraite les données du profileur avec découpage temporel
    
    Intrants:
    - data_dict : dict - Données depuis load_profiler_data()
    - start_time : str ou datetime - Temps de début d'analyse
    - duration_minutes : float - Durée d'analyse en minutes
    
    Extrants:
    - processed_data : dict - Données prétraitées avec indices temporels filtrés
    - start_timestamp : datetime - Timestamp de début
    - end_timestamp : datetime - Timestamp de fin
    
    Format des intrants:
    - start_time: "YYYY-MM-DD HH:MM:SS.ffffff" ou datetime
    - duration_minutes > 0
    '''
    if isinstance(start_time, str):
        start_timestamp = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
    else:
        start_timestamp = start_time
    
    end_timestamp = start_timestamp + timedelta(minutes=duration_minutes)
    
    # Trouver les indices temporels dans la plage
    timestamps = data_dict['timestamps']
    time_mask = [(t >= start_timestamp) and (t <= end_timestamp) for t in timestamps]
    valid_indices = np.where(time_mask)[0]
    
    processed_data = {
        'timestamps': [timestamps[i] for i in valid_indices],
        'range_data': data_dict['range_data'],
        'config': data_dict['config'],
        'velocity_profiles': {},
        'snr_data': {},
        'correlation_data': {}
    }
    
    # Filtrage temporel des profils de vitesse
    for vel_comp, profiles in data_dict['velocity_profiles'].items():
        processed_data['velocity_profiles'][vel_comp] = profiles[valid_indices, :]
    
    # Filtrage temporel des données SNR et corrélation
    for beam, snr_data in data_dict['snr_data'].items():
        processed_data['snr_data'][beam] = snr_data[valid_indices, :]
    
    for beam, cor_data in data_dict['correlation_data'].items():
        processed_data['correlation_data'][beam] = cor_data[valid_indices, :]
    
    return processed_data, start_timestamp, end_timestamp

def filter_profiler_data_quality(processed_data, snr_threshold=9, correlation_threshold=70):
    '''
    Filtre les données du profileur selon les critères de qualité SNR et corrélation
    
    Intrants:
    - processed_data : dict - Données depuis preprocess_profiler_data()
    - snr_threshold : float - Seuil minimum SNR (défaut: 9)
    - correlation_threshold : float - Seuil minimum corrélation en % (défaut: 70)
    
    Extrants:
    - filtered_data : dict - Données filtrées avec NaN pour points de mauvaise qualité
    
    Format des intrants:
    - processed_data avec structure standard
    - Seuils typiques: SNR ≥ 9, Corrélation ≥ 70%
    '''
    filtered_data = processed_data.copy()
    filtered_data['velocity_profiles'] = {}
    
    # Pour chaque composante de vitesse
    for vel_comp, vel_profiles in processed_data['velocity_profiles'].items():
        filtered_profiles = vel_profiles.copy()
        n_times, n_cells = vel_profiles.shape
        
        # Pour chaque cellule de mesure
        for cell_idx in range(n_cells):
            # Critères de qualité pour tous les faisceaux
            quality_mask = np.ones(n_times, dtype=bool)
            
            # Vérification SNR pour tous les faisceaux
            for beam, snr_data in processed_data['snr_data'].items():
                if snr_data.shape[1] > cell_idx:
                    quality_mask &= (snr_data[:, cell_idx] >= snr_threshold)
            
            # Vérification corrélation pour tous les faisceaux
            for beam, cor_data in processed_data['correlation_data'].items():
                if cor_data.shape[1] > cell_idx:
                    quality_mask &= (cor_data[:, cell_idx] >= correlation_threshold)
            
            # Application du masque qualité
            filtered_profiles[~quality_mask, cell_idx] = np.nan
        
        filtered_data['velocity_profiles'][vel_comp] = filtered_profiles
    
    return filtered_data

def calculate_velocity_magnitude(filtered_data):
    '''
    Calcule la magnitude de vitesse à partir des composantes X, Y, Z
    
    Intrants:
    - filtered_data : dict - Données filtrées avec composantes de vitesse
    
    Extrants:
    - velocity_magnitude : numpy array - Magnitude de vitesse par cellule et temps
    
    Format des intrants:
    - filtered_data doit contenir VelX, VelY, VelZ1 dans velocity_profiles
    '''
    vel_x = filtered_data['velocity_profiles'].get('VelX', 0)
    vel_y = filtered_data['velocity_profiles'].get('VelY', 0)
    vel_z = filtered_data['velocity_profiles'].get('VelZ1', 0)
    
    velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
    return velocity_magnitude

def calculate_profile_statistics(filtered_data):
    '''
    Calcule les statistiques des profils de vitesse par cellule
    
    Intrants:
    - filtered_data : dict - Données filtrées du profileur
    
    Extrants:
    - profile_stats : dict - Statistiques par composante et par cellule
        - mean_profiles : profils moyens temporels
        - std_profiles : écarts-types par cellule
        - max_profiles : maxima par cellule
        - min_profiles : minima par cellule
    
    Format des intrants:
    - filtered_data avec velocity_profiles contenant arrays (n_times, n_cells)
    '''
    profile_stats = {
        'mean_profiles': {},
        'std_profiles': {},
        'max_profiles': {},
        'min_profiles': {},
        'rms_profiles': {}
    }
    
    # Calcul de la magnitude
    velocity_magnitude = calculate_velocity_magnitude(filtered_data)
    all_components = dict(filtered_data['velocity_profiles'])
    all_components['Magnitude'] = velocity_magnitude
    
    for vel_comp, profiles in all_components.items():
        # Ignorer les NaN dans les calculs
        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore", category=RuntimeWarning)
            
            profile_stats['mean_profiles'][vel_comp] = np.nanmean(profiles, axis=0)
            profile_stats['std_profiles'][vel_comp] = np.nanstd(profiles, axis=0)
            profile_stats['max_profiles'][vel_comp] = np.nanmax(profiles, axis=0)
            profile_stats['min_profiles'][vel_comp] = np.nanmin(profiles, axis=0)
            profile_stats['rms_profiles'][vel_comp] = np.sqrt(np.nanmean(profiles**2, axis=0))
    
    return profile_stats

def calculate_shear_profiles(filtered_data, cell_distances):
    '''
    Calcule les profils de cisaillement (gradients de vitesse)
    
    Intrants:
    - filtered_data : dict - Données filtrées du profileur
    - cell_distances : array - Distances/profondeurs des cellules (m)
    
    Extrants:
    - shear_profiles : dict - Gradients de vitesse par composante
        - dU_dz, dV_dz, dW_dz : gradients par composante (s⁻¹)
        - shear_rate : taux de cisaillement total
    
    Format des intrants:
    - cell_distances ordonnées selon l'axe de mesure du profileur
    - Espacement régulier recommandé entre cellules
    '''
    shear_profiles = {}
    
    for vel_comp, profiles in filtered_data['velocity_profiles'].items():
        n_times, n_cells = profiles.shape
        gradients = np.full((n_times, n_cells-1), np.nan)
        
        # Calcul des gradients entre cellules adjacentes
        for t in range(n_times):
            for c in range(n_cells-1):
                if not (np.isnan(profiles[t, c]) or np.isnan(profiles[t, c+1])):
                    dv = profiles[t, c+1] - profiles[t, c]
                    dz = cell_distances[c+1] - cell_distances[c]
                    if abs(dz) > 1e-10:
                        gradients[t, c] = dv / dz
        
        gradient_name = f'd{vel_comp[-1]}_dz' if vel_comp.endswith('1') else f'd{vel_comp[-1]}_dz'
        shear_profiles[gradient_name] = gradients
    
    # Calcul du taux de cisaillement total si on a les 3 composantes
    if all(comp in filtered_data['velocity_profiles'] for comp in ['VelX', 'VelY', 'VelZ1']):
        du_dz = shear_profiles.get('dX_dz', 0)
        dv_dz = shear_profiles.get('dY_dz', 0)
        dw_dz = shear_profiles.get('dZ_dz', 0)
        
        shear_profiles['shear_rate'] = np.sqrt(du_dz**2 + dv_dz**2 + dw_dz**2)
    
    return shear_profiles

def calculate_turbulence_profiles(filtered_data):
    '''
    Calcule les profils de turbulence par cellule (contraintes de Reynolds)
    
    Intrants:
    - filtered_data : dict - Données filtrées du profileur
    
    Extrants:
    - turbulence_profiles : dict - Paramètres de turbulence par cellule
        - reynolds_stress_profiles : contraintes de Reynolds
        - tke_profiles : énergie cinétique turbulente par cellule
        - turbulence_intensity_profiles : intensité turbulente par composante
    
    Format des intrants:
    - filtered_data avec données à fréquence constante pour analyse turbulente
    '''
    turbulence_profiles = {
        'reynolds_stress_profiles': {},
        'tke_profiles': {},
        'turbulence_intensity_profiles': {}
    }
    
    velocity_components = ['VelX', 'VelY', 'VelZ1']
    available_components = [comp for comp in velocity_components 
                          if comp in filtered_data['velocity_profiles']]
    
    if len(available_components) < 2:
        return turbulence_profiles
    
    # Extraction des composantes disponibles
    vel_data = {}
    for comp in available_components:
        vel_data[comp] = filtered_data['velocity_profiles'][comp]
    
    n_times, n_cells = list(vel_data.values())[0].shape
    
    # Calcul par cellule
    for cell_idx in range(n_cells):
        cell_data = {}
        cell_means = {}
        
        # Extraction des données de la cellule et calcul des moyennes
        for comp in available_components:
            cell_values = vel_data[comp][:, cell_idx]
            valid_mask = ~np.isnan(cell_values)
            
            if np.sum(valid_mask) > 10:  # Minimum de points valides
                cell_data[comp] = cell_values[valid_mask]
                cell_means[comp] = np.mean(cell_data[comp])
            else:
                cell_data[comp] = np.array([])
                cell_means[comp] = np.nan
        
        # Calcul des fluctuations et contraintes de Reynolds
        if len(cell_data['VelX']) > 0 and len(cell_data['VelY']) > 0:
            u_prime = cell_data['VelX'] - cell_means['VelX']
            v_prime = cell_data['VelY'] - cell_means['VelY']
            
            # Tronquer à la longueur minimale
            min_length = min(len(u_prime), len(v_prime))
            u_prime = u_prime[:min_length]
            v_prime = v_prime[:min_length]
            
            # Contraintes de Reynolds
            reynolds_stress = np.mean(u_prime * v_prime)
            turbulence_profiles['reynolds_stress_profiles'][f'Cell_{cell_idx+1}'] = reynolds_stress
            
            # Énergie cinétique turbulente
            u_var = np.var(u_prime)
            v_var = np.var(v_prime)
            
            if 'VelZ1' in cell_data and len(cell_data['VelZ1']) > 0:
                w_prime = cell_data['VelZ1'] - cell_means['VelZ1']
                w_prime = w_prime[:min_length]
                w_var = np.var(w_prime)
                tke = 0.5 * (u_var + v_var + w_var)
            else:
                tke = 0.5 * (u_var + v_var)  # TKE 2D
            
            turbulence_profiles['tke_profiles'][f'Cell_{cell_idx+1}'] = tke
            
            # Intensités turbulentes
            for comp in available_components:
                if comp in cell_means and not np.isnan(cell_means[comp]):
                    if abs(cell_means[comp]) > 1e-6:
                        comp_var = np.var(cell_data[comp][:min_length])
                        intensity = np.sqrt(comp_var) / abs(cell_means[comp])
                        turbulence_profiles['turbulence_intensity_profiles'][f'{comp}_Cell_{cell_idx+1}'] = intensity
    
    return turbulence_profiles

def extract_time_series_by_cell(filtered_data, cell_index, resampling_interval='1S'):
    '''
    Extrait les séries temporelles pour une cellule spécifique avec rééchantillonnage
    
    Intrants:
    - filtered_data : dict - Données filtrées du profileur
    - cell_index : int - Indice de la cellule (0-based)
    - resampling_interval : str - Intervalle de rééchantillonnage pandas (défaut: '1S')
    
    Extrants:
    - cell_time_series : pandas DataFrame - Séries temporelles de la cellule
        - colonnes: Time, VelX, VelY, VelZ1, Magnitude
    
    Format des intrants:
    - cell_index < nombre de cellules disponibles
    - resampling_interval: '1S', '0.5S', '2S', etc.
    '''
    timestamps = filtered_data['timestamps']
    
    # Création du DataFrame avec timestamp
    df_data = {'Time': timestamps}
    
    # Ajout des composantes de vitesse
    for vel_comp, profiles in filtered_data['velocity_profiles'].items():
        if profiles.shape[1] > cell_index:
            df_data[vel_comp] = profiles[:, cell_index]
    
    # Calcul de la magnitude si on a les composantes
    if all(comp in df_data for comp in ['VelX', 'VelY', 'VelZ1']):
        df_data['Magnitude'] = np.sqrt(df_data['VelX']**2 + df_data['VelY']**2 + df_data['VelZ1']**2)
    
    # Création du DataFrame
    cell_time_series = pd.DataFrame(df_data)
    cell_time_series.set_index('Time', inplace=True)
    
    # Rééchantillonnage si demandé
    if resampling_interval:
        cell_time_series = cell_time_series.resample(resampling_interval).mean()
    
    return cell_time_series

def calculate_bottom_boundary_layer(profile_stats, cell_distances, reference_height=0.01):
    '''
    Analyse de la couche limite près du fond à partir des profils moyens
    
    Intrants:
    - profile_stats : dict - Statistiques des profils depuis calculate_profile_statistics()
    - cell_distances : array - Distances/hauteurs des cellules (m)
    - reference_height : float - Hauteur de référence pour calculs (m) (défaut: 0.01)
    
    Extrants:
    - boundary_layer_params : dict - Paramètres de couche limite
        - friction_velocity : vitesse de friction u* (m/s)
        - roughness_length : longueur de rugosité z0 (m)
        - boundary_layer_thickness : épaisseur couche limite (m)
        - velocity_at_reference : vitesse à hauteur de référence (m/s)
    
    Format des intrants:
    - cell_distances ordonnées depuis le fond (z=0 au fond)
    - reference_height typiquement 0.01-0.05 m pour études côtières
    '''
    boundary_layer_params = {}
    
    # Utiliser la magnitude ou VelX comme vitesse principale
    if 'Magnitude' in profile_stats['mean_profiles']:
        velocity_profile = profile_stats['mean_profiles']['Magnitude']
    elif 'VelX' in profile_stats['mean_profiles']:
        velocity_profile = profile_stats['mean_profiles']['VelX']
    else:
        return boundary_layer_params
    
    # Masquer les valeurs NaN et les hauteurs nulles
    valid_mask = ~np.isnan(velocity_profile) & (cell_distances > 0) & (velocity_profile > 0)
    
    if np.sum(valid_mask) < 3:
        return boundary_layer_params
    
    z_valid = cell_distances[valid_mask]
    u_valid = velocity_profile[valid_mask]
    
    # Ajustement profil logarithmique: u = (u*/κ) * ln(z/z0)
    # où κ = 0.41 (constante de von Kármán)
    kappa = 0.41
    
    try:
        # Régression linéaire sur ln(z) vs u
        ln_z = np.log(z_valid)
        coeffs = np.polyfit(ln_z, u_valid, 1)
        
        # Extraction des paramètres
        u_star = coeffs[0] * kappa  # Vitesse de friction
        z0 = np.exp(-coeffs[1] / coeffs[0])  # Longueur de rugosité
        
        # Vitesse à la hauteur de référence
        if reference_height > z0:
            u_ref = (u_star / kappa) * np.log(reference_height / z0)
        else:
            u_ref = np.nan
        
        # Épaisseur de couche limite (approximation)
        max_height = np.max(z_valid)
        boundary_layer_thickness = max_height  # Simplification
        
        boundary_layer_params = {
            'friction_velocity': u_star,
            'roughness_length': z0,
            'boundary_layer_thickness': boundary_layer_thickness,
            'velocity_at_reference': u_ref,
            'reference_height': reference_height,
            'von_karman_constant': kappa,
            'fit_quality_r2': np.corrcoef(ln_z, u_valid)[0,1]**2
        }
        
    except (np.linalg.LinAlgError, ValueError):
        # En cas d'échec de l'ajustement
        boundary_layer_params = {
            'friction_velocity': np.nan,
            'roughness_length': np.nan,
            'boundary_layer_thickness': np.nan,
            'velocity_at_reference': np.nan
        }
    
    return boundary_layer_params

def analyze_profiler_data_complete(data_dict, start_time, duration_minutes, 
                                 snr_threshold=9, correlation_threshold=70):
    '''
    Analyse complète des données du profileur acoustique
    
    Intrants:
    - data_dict : dict - Données brutes depuis load_profiler_data()
    - start_time : str ou datetime - Temps de début d'analyse
    - duration_minutes : float - Durée d'analyse en minutes
    - snr_threshold : float - Seuil SNR (défaut: 9)
    - correlation_threshold : float - Seuil corrélation (défaut: 70)
    
    Extrants:
    - complete_analysis : dict - Analyse complète
        - filtered_data : données filtrées
        - profile_statistics : statistiques par cellule
        - shear_profiles : gradients de vitesse
        - turbulence_profiles : paramètres turbulents
        - boundary_layer : paramètres de couche limite
        - data_quality : métriques de qualité
    
    Format des intrants:
    - data_dict avec structure standard de load_profiler_data()
    - Paramètres de seuils selon conditions expérimentales
    '''
    
    # 1. Prétraitement temporel
    processed_data, start_timestamp, end_timestamp = preprocess_profiler_data(
        data_dict, start_time, duration_minutes)
    
    # 2. Filtrage qualité
    filtered_data = filter_profiler_data_quality(
        processed_data, snr_threshold, correlation_threshold)
    
    # 3. Calcul des distances des cellules
    if len(filtered_data['range_data']) > 0:
        cell_distances = np.abs(filtered_data['range_data'])  # Distances absolues
    else:
        # Distances par défaut si non disponibles
        n_cells = list(filtered_data['velocity_profiles'].values())[0].shape[1]
        cell_distances = np.arange(n_cells) * 0.001  # 1 mm par cellule par défaut
    
    # 4. Statistiques des profils
    profile_stats = calculate_profile_statistics(filtered_data)
    
    # 5. Analyse des cisaillements
    shear_profiles = calculate_shear_profiles(filtered_data, cell_distances)
    
    # 6. Analyse de turbulence
    turbulence_profiles = calculate_turbulence_profiles(filtered_data)
    
    # 7. Analyse de couche limite
    boundary_layer_params = calculate_bottom_boundary_layer(profile_stats, cell_distances)
    
    # 8. Métriques de qualité des données
    total_points = len(filtered_data['timestamps']) * len(cell_distances)
    valid_points = 0
    
    for vel_comp, profiles in filtered_data['velocity_profiles'].items():
        valid_points += np.sum(~np.isnan(profiles))
    
    data_quality = {
        'measurement_duration_s': (end_timestamp - start_timestamp).total_seconds(),
        'number_of_cells': len(cell_distances),
        'total_data_points': total_points,
        'valid_data_points': valid_points,
        'data_completeness_percent': (valid_points / total_points) * 100 if total_points > 0 else 0,
        'snr_threshold_used': snr_threshold,
        'correlation_threshold_used': correlation_threshold,
        'cell_distances': cell_distances.tolist()
    }
    
    # 9. Compilation des résultats
    complete_analysis = {
        'filtered_data': filtered_data,
        'profile_statistics': profile_stats,
        'shear_profiles': shear_profiles,
        'turbulence_profiles': turbulence_profiles,
        'boundary_layer': boundary_layer_params,
        'data_quality': data_quality,
        'time_range': {
            'start_time': start_timestamp,
            'end_time': end_timestamp
        }
    }
    
    return complete_analysis

def export_profiler_results_to_dataframes(complete_analysis):
    '''
    Exporte les résultats d'analyse du profileur vers des DataFrames
    
    Intrants:
    - complete_analysis : dict - Résultats depuis analyze_profiler_data_complete()
    
    Extrants:
    - summary_df : pandas DataFrame - Résumé des paramètres principaux
    - profile_stats_df : pandas DataFrame - Statistiques par cellule
    - turbulence_df : pandas DataFrame - Paramètres turbulents
    - boundary_layer_df : pandas DataFrame - Paramètres de couche limite
    
    Format des intrants:
    - complete_analysis avec structure standard
    '''
    
    # DataFrame résumé
    cell_distances = complete_analysis['data_quality']['cell_distances']
    profile_stats = complete_analysis['profile_statistics']
    
    summary_data = []
    
    # Paramètres globaux
    summary_data.extend([
        ['Durée mesure (s)', complete_analysis['data_quality']['measurement_duration_s']],
        ['Nombre de cellules', complete_analysis['data_quality']['number_of_cells']],
        ['Complétude données (%)', complete_analysis['data_quality']['data_completeness_percent']],
        ['Vitesse de friction u* (m/s)', complete_analysis['boundary_layer'].get('friction_velocity', np.nan)],
        ['Longueur rugosité z0 (m)', complete_analysis['boundary_layer'].get('roughness_length', np.nan)]
    ])
    
    summary_df = pd.DataFrame(summary_data, columns=['Paramètre', 'Valeur'])
    
    # DataFrame des statistiques par cellule
    profile_data = []
    
    for i, distance in enumerate(cell_distances):
        row_data = {'Cell_Index': i+1, 'Distance_m': distance}
        
        for stat_type in ['mean_profiles', 'std_profiles', 'max_profiles']:
            for vel_comp in profile_stats[stat_type]:
                if i < len(profile_stats[stat_type][vel_comp]):
                    col_name = f'{stat_type}_{vel_comp}'
                    row_data[col_name] = profile_stats[stat_type][vel_comp][i]
        
        profile_data.append(row_data)
    
    profile_stats_df = pd.DataFrame(profile_data)
    
    # DataFrame de turbulence
    turbulence_data = []
    turb_profiles = complete_analysis['turbulence_profiles']
    
    for param_type in ['reynolds_stress_profiles', 'tke_profiles']:
        if param_type in turb_profiles:
            for cell_name, value in turb_profiles[param_type].items():
                turbulence_data.append({
                    'Cell': cell_name,
                    'Parameter': param_type,
                    'Value': value
                })
    
    turbulence_df = pd.DataFrame(turbulence_data)
    
    # DataFrame couche limite
    boundary_layer_data = []
    bl_params = complete_analysis['boundary_layer']
    
    for param, value in bl_params.items():
        boundary_layer_data.append({'Parameter': param, 'Value': value})
    
    boundary_layer_df = pd.DataFrame(boundary_layer_data)
    
    return summary_df, profile_stats_df, turbulence_df, boundary_layer_df