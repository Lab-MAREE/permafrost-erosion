# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 18:04:51 2025

@author: Hatim.BEN_SAID
"""

"""
Fonctions de calcul des propriétés thermiques des sols gelés
Basé sur les modèles de Côté & Konrad (2005) et Kersten (1949)
Adapté pour l'étude du pergélisol en modélisation physique
"""

import numpy as np
import pandas as pd
import math

def calculate_basic_soil_properties(density_dry, density_wet, water_content, specific_gravity=2.65):
    '''
    Calcule les propriétés physiques de base d'un sol
    
    Intrants:
    - density_dry : float - Masse volumique sèche (kg/m³)
    - density_wet : float - Masse volumique humide (kg/m³)
    - water_content : float - Teneur en eau massique (décimal, ex: 0.15 pour 15%)
    - specific_gravity : float - Poids spécifique des grains solides (défaut: 2.65)
    
    Extrants:
    - properties : dict - Dictionnaire des propriétés calculées
        - porosity : porosité (-)
        - void_ratio : indice des vides (-)
        - saturation_degree : degré de saturation (-)
        - volumetric_water_content : teneur en eau volumique (-)
        - solid_density : masse volumique des grains solides (kg/m³)
    
    Format des intrants:
    - density_dry, density_wet > 0
    - water_content entre 0 et 1 (décimal)
    - specific_gravity typiquement entre 2.6-2.8 pour sols naturels
    '''
    rho_w = 1000  # Masse volumique de l'eau (kg/m³)
    
    # Masse volumique des grains solides
    rho_s = specific_gravity * rho_w
    
    # Porosité
    n = 1 - (density_dry / rho_s)
    
    # Indice des vides
    e = n / (1 - n)
    
    # Teneur en eau volumique
    theta = water_content * specific_gravity * (1 - n)
    
    # Degré de saturation
    Sr = theta / n if n > 0 else 0
    
    properties = {
        'porosity': n,
        'void_ratio': e,
        'saturation_degree': Sr,
        'volumetric_water_content': theta,
        'solid_density': rho_s,
        'water_density': rho_w
    }
    
    return properties

def calculate_unfrozen_water_content(density_dry, specific_surface, temperature):
    '''
    Calcule la teneur en eau non gelée selon le modèle d'Anderson et Tice
    
    Intrants:
    - density_dry : float - Masse volumique sèche (kg/m³)
    - specific_surface : float - Surface spécifique des particules (m²/g)
    - temperature : float - Température du sol gelé (°C, négative)
    
    Extrants:
    - theta_u : float - Teneur en eau non gelée volumique (-)
    
    Format des intrants:
    - density_dry > 0
    - specific_surface > 0 (typiquement 1-100 m²/g selon le type de sol)
    - temperature < 0 (sol gelé)
    '''
    if temperature >= 0:
        return 0.0
    
    # Modèle empirique d'Anderson et Tice (Andersland & Anderson, 1978)
    ln_theta_u = (np.log(density_dry) + 0.5519 * np.log(specific_surface) - 
                  1.449 * np.log(-temperature) * (specific_surface**(-0.264)) - 11.25)
    
    theta_u = np.exp(ln_theta_u)
    
    return max(0, theta_u)

def calculate_heat_capacities(porosity, saturation_degree, theta_u=0, 
                            c_solid=2.0, c_water=4.18, c_ice=1.90, c_air=0.0):
    '''
    Calcule les capacités calorifiques du sol gelé et non gelé
    
    Intrants:
    - porosity : float - Porosité du sol (-)
    - saturation_degree : float - Degré de saturation (-)
    - theta_u : float - Teneur en eau non gelée volumique (-) (défaut: 0)
    - c_solid : float - Capacité calorifique des solides (MJ/m³·°C) (défaut: 2.0)
    - c_water : float - Capacité calorifique de l'eau (MJ/m³·°C) (défaut: 4.18)
    - c_ice : float - Capacité calorifique de la glace (MJ/m³·°C) (défaut: 1.90)
    - c_air : float - Capacité calorifique de l'air (MJ/m³·°C) (défaut: 0.0)
    
    Extrants:
    - C_frozen : float - Capacité calorifique du sol gelé (MJ/m³·°C)
    - C_unfrozen : float - Capacité calorifique du sol non gelé (MJ/m³·°C)
    
    Format des intrants:
    - Tous les paramètres entre 0 et 1 pour les fractions volumiques
    - Capacités calorifiques en MJ/m³·°C selon standards géotechniques
    '''
    n = porosity
    Sr = saturation_degree
    
    # Capacité calorifique du sol gelé (équation 15)
    C_frozen = (c_solid * (1 - n) + 
                c_ice * (Sr * n - theta_u) + 
                c_water * theta_u + 
                c_air * (1 - Sr) * n)
    
    # Capacité calorifique du sol non gelé (équation 16)
    C_unfrozen = (c_solid * (1 - n) + 
                  c_water * Sr * n + 
                  c_air * (1 - Sr) * n)
    
    return C_frozen, C_unfrozen

def calculate_latent_heat(saturation_degree, porosity, theta_u=0, L_water=333):
    '''
    Calcule la chaleur latente de fusion volumique du sol
    
    Intrants:
    - saturation_degree : float - Degré de saturation (-)
    - porosity : float - Porosité du sol (-)
    - theta_u : float - Teneur en eau non gelée volumique (-) (défaut: 0)
    - L_water : float - Chaleur latente de fusion de l'eau (MJ/m³) (défaut: 333)
    
    Extrants:
    - L_soil : float - Chaleur latente de fusion volumique du sol (MJ/m³)
    
    Format des intrants:
    - saturation_degree, porosity entre 0 et 1
    - L_water = 333 MJ/m³ pour l'eau pure
    '''
    # Équation 17: L = Lw * (Sr * n - θu)
    L_soil = L_water * (saturation_degree * porosity - theta_u)
    
    return max(0, L_soil)

def calculate_solid_thermal_conductivity(quartz_fraction):
    '''
    Calcule la conductivité thermique de la phase solide selon Côté & Konrad (2005)
    
    Intrants:
    - quartz_fraction : float - Fraction volumique de quartz (-) entre 0 et 1
    
    Extrants:
    - lambda_s : float - Conductivité thermique des solides (W/m·°C)
    
    Format des intrants:
    - quartz_fraction entre 0 et 1 (0.2 = 20% de quartz)
    - Peut être estimé depuis la limite liquide avec: q = 2.8 * wL^(-0.77)
    '''
    q = quartz_fraction
    
    # Équation 20 du modèle de Côté & Konrad (2005)
    if q < 0.2:
        lambda_s = (7.7**q) * (3.0**(1-q))
    else:
        lambda_s = (7.7**q) * (2.0**(1-q))
    
    return lambda_s

def calculate_dry_thermal_conductivity(lambda_s, porosity, material_type='angular'):
    '''
    Calcule la conductivité thermique du matériau sec selon Côté & Konrad (2005)
    
    Intrants:
    - lambda_s : float - Conductivité thermique des solides (W/m·°C)
    - porosity : float - Porosité du sol (-)
    - material_type : str - Type de matériau ('angular', 'crushed', 'cemented')
    
    Extrants:
    - lambda_dry : float - Conductivité thermique du sol sec (W/m·°C)
    
    Format des intrants:
    - lambda_s typiquement entre 2-8 W/m·°C pour sols naturels
    - porosity entre 0 et 1
    - material_type détermine le paramètre beta structural
    '''
    lambda_air = 0.024  # Conductivité thermique de l'air (W/m·°C)
    n = porosity
    
    # Paramètre structural beta selon le type de matériau
    beta_values = {
        'natural': 0.81,
        'angular': 0.54,
        'crushed': 0.54,
        'cemented': 0.34
    }
    
    beta = beta_values.get(material_type, 0.54)
    
    # Ratio des conductivités
    ratio = lambda_air / lambda_s
    
    # Si le ratio > 0.15, utiliser beta = 0.46
    if ratio > 0.15:
        beta = 0.46
    
    # Paramètre structural κ2P (équation 22)
    kappa_2P = 0.29 * ((15 * ratio)**beta)
    
    # Conductivité thermique sèche (équation 21)
    lambda_dry = ((kappa_2P * lambda_s - lambda_air) * (1 - n) + lambda_air) / (1 + (kappa_2P - 1) * (1 - n))
    
    return lambda_dry

def calculate_saturated_thermal_conductivity(lambda_s, porosity, saturation_degree, 
                                           theta_u=0, lambda_water=0.6, lambda_ice=2.24):
    '''
    Calcule les conductivités thermiques saturées (gelée et non gelée)
    
    Intrants:
    - lambda_s : float - Conductivité thermique des solides (W/m·°C)
    - porosity : float - Porosité du sol (-)
    - saturation_degree : float - Degré de saturation (-)
    - theta_u : float - Teneur en eau non gelée volumique (-) (défaut: 0)
    - lambda_water : float - Conductivité thermique de l'eau (W/m·°C) (défaut: 0.6)
    - lambda_ice : float - Conductivité thermique de la glace (W/m·°C) (défaut: 2.24)
    
    Extrants:
    - lambda_sat_unfrozen : float - Conductivité saturée non gelée (W/m·°C)
    - lambda_sat_frozen : float - Conductivité saturée gelée (W/m·°C)
    
    Format des intrants:
    - Tous les paramètres positifs
    - theta_u <= saturation_degree * porosity
    '''
    n = porosity
    Sr = saturation_degree
    
    # Conductivité saturée non gelée (équation 23)
    lambda_sat_unfrozen = (lambda_s**(1-n)) * (lambda_water**n)
    
    # Conductivité saturée gelée (équation 24)
    lambda_sat_frozen = ((lambda_s**(1-n)) * 
                        (lambda_ice**(Sr*n - theta_u)) * 
                        (lambda_water**theta_u))
    
    return lambda_sat_unfrozen, lambda_sat_frozen

def get_kappa_values(soil_type):
    '''
    Retourne les valeurs du paramètre κ selon le type de sol (Côté & Konrad, 2005)
    
    Intrants:
    - soil_type : str - Type de sol ('gravel_sand', 'fine_sand', 'silt_clay', 'organic')
    
    Extrants:
    - kappa_u : float - Paramètre κ pour l'état non gelé
    - kappa_f : float - Paramètre κ pour l'état gelé
    
    Format des intrants:
    - soil_type doit correspondre aux catégories définies
    '''
    kappa_values = {
        'gravel_sand': {'unfrozen': 4.60, 'frozen': 1.70},      # Gravier et sables grenus
        'fine_sand': {'unfrozen': 3.55, 'frozen': 0.95},       # Sables fins et moyens
        'silt_clay': {'unfrozen': 1.90, 'frozen': 0.85},       # Sols limoneux et argileux
        'organic': {'unfrozen': 0.60, 'frozen': 0.25}          # Sols à fibres organiques
    }
    
    if soil_type not in kappa_values:
        # Valeur par défaut pour sables fins
        soil_type = 'fine_sand'
    
    return kappa_values[soil_type]['unfrozen'], kappa_values[soil_type]['frozen']

def calculate_thermal_conductivity_cote_konrad(lambda_dry, lambda_sat_unfrozen, lambda_sat_frozen,
                                             saturation_degree, soil_type):
    '''
    Calcule la conductivité thermique selon le modèle de Côté & Konrad (2005)
    
    Intrants:
    - lambda_dry : float - Conductivité thermique sèche (W/m·°C)
    - lambda_sat_unfrozen : float - Conductivité saturée non gelée (W/m·°C)
    - lambda_sat_frozen : float - Conductivité saturée gelée (W/m·°C)
    - saturation_degree : float - Degré de saturation (-)
    - soil_type : str - Type de sol pour paramètre κ
    
    Extrants:
    - lambda_unfrozen : float - Conductivité thermique non gelée (W/m·°C)
    - lambda_frozen : float - Conductivité thermique gelée (W/m·°C)
    
    Format des intrants:
    - Conductivités thermiques > 0
    - saturation_degree entre 0 et 1
    - soil_type selon get_kappa_values()
    '''
    Sr = saturation_degree
    kappa_u, kappa_f = get_kappa_values(soil_type)
    
    # Conductivité thermique non gelée (équation 27)
    lambda_unfrozen = ((kappa_u * lambda_sat_unfrozen - lambda_dry) * Sr + lambda_dry) / (1 + (kappa_u - 1) * Sr)
    
    # Conductivité thermique gelée (équation 27)
    lambda_frozen = ((kappa_f * lambda_sat_frozen - lambda_dry) * Sr + lambda_dry) / (1 + (kappa_f - 1) * Sr)
    
    return lambda_unfrozen, lambda_frozen

def calculate_thermal_conductivity_kersten(density_dry, water_content, soil_type='fine'):
    '''
    Calcule la conductivité thermique selon le modèle empirique de Kersten (1949)
    Utilisé comme méthode de vérification
    
    Intrants:
    - density_dry : float - Masse volumique sèche (kg/m³)
    - water_content : float - Teneur en eau massique (décimal)
    - soil_type : str - Type de sol ('fine' ou 'coarse')
    
    Extrants:
    - lambda_unfrozen_k : float - Conductivité non gelée Kersten (W/m·°C)
    - lambda_frozen_k : float - Conductivité gelée Kersten (W/m·°C)
    
    Format des intrants:
    - density_dry en kg/m³ (typiquement 1200-2000)
    - water_content en décimal (ex: 0.15 pour 15%)
    - soil_type: 'fine' pour limons/argiles, 'coarse' pour sables/graviers
    '''
    rho_d = density_dry / 1000  # Conversion en g/cm³ pour les formules de Kersten
    w_percent = water_content * 100  # Conversion en pourcentage
    
    if soil_type == 'fine':
        # Sols fins (équations 34-35)
        lambda_unfrozen_k = 0.1442 * (0.9 * np.log10(w_percent) - 0.2) * (10**(0.6243 * rho_d))
        lambda_frozen_k = (0.001442 * (10**(1.373 * rho_d)) + 
                          0.01226 * (10**(0.4994 * rho_d)) * w_percent)
    else:
        # Sols grenus (équations 36-37)
        lambda_unfrozen_k = 0.1442 * (0.7 * np.log10(w_percent) + 0.4) * (10**(0.6243 * rho_d))
        lambda_frozen_k = (0.01096 * (10**(0.8116 * rho_d)) + 
                          0.00461 * (10**(0.9115 * rho_d)) * w_percent)
    
    return lambda_unfrozen_k, lambda_frozen_k

def analyze_thermal_properties_complete(density_dry, density_wet, water_content, 
                                      quartz_fraction=0.2, soil_type='fine_sand',
                                      material_type='angular', specific_gravity=2.65,
                                      temperature=-5.0, specific_surface=10.0):
    '''
    Analyse complète des propriétés thermiques d'un sol gelé
    
    Intrants:
    - density_dry : float - Masse volumique sèche (kg/m³)
    - density_wet : float - Masse volumique humide (kg/m³)
    - water_content : float - Teneur en eau massique (décimal)
    - quartz_fraction : float - Fraction de quartz (-) (défaut: 0.2)
    - soil_type : str - Type de sol pour κ (défaut: 'fine_sand')
    - material_type : str - Type structural (défaut: 'angular')
    - specific_gravity : float - Poids spécifique (défaut: 2.65)
    - temperature : float - Température pour eau non gelée (°C) (défaut: -5.0)
    - specific_surface : float - Surface spécifique (m²/g) (défaut: 10.0)
    
    Extrants:
    - results : dict - Dictionnaire complet des propriétés thermiques calculées
    
    Format des intrants:
    - density_dry, density_wet en kg/m³
    - water_content en décimal (0.15 = 15%)
    - Autres paramètres selon fonctions individuelles
    '''
    
    # 1. Propriétés physiques de base
    basic_props = calculate_basic_soil_properties(density_dry, density_wet, water_content, specific_gravity)
    
    # 2. Teneur en eau non gelée (si température < 0)
    if temperature < 0:
        theta_u = calculate_unfrozen_water_content(density_dry, specific_surface, temperature)
    else:
        theta_u = basic_props['volumetric_water_content']
    
    # 3. Capacités calorifiques
    C_frozen, C_unfrozen = calculate_heat_capacities(
        basic_props['porosity'], 
        basic_props['saturation_degree'], 
        theta_u
    )
    
    # 4. Chaleur latente de fusion
    L_soil = calculate_latent_heat(
        basic_props['saturation_degree'], 
        basic_props['porosity'], 
        theta_u
    )
    
    # 5. Conductivités thermiques selon Côté & Konrad
    lambda_s = calculate_solid_thermal_conductivity(quartz_fraction)
    lambda_dry = calculate_dry_thermal_conductivity(lambda_s, basic_props['porosity'], material_type)
    
    lambda_sat_u, lambda_sat_f = calculate_saturated_thermal_conductivity(
        lambda_s, basic_props['porosity'], basic_props['saturation_degree'], theta_u
    )
    
    lambda_u, lambda_f = calculate_thermal_conductivity_cote_konrad(
        lambda_dry, lambda_sat_u, lambda_sat_f, 
        basic_props['saturation_degree'], soil_type
    )
    
    # 6. Vérification avec Kersten (1949)
    kersten_type = 'fine' if soil_type in ['silt_clay', 'organic'] else 'coarse'
    lambda_u_kersten, lambda_f_kersten = calculate_thermal_conductivity_kersten(
        density_dry, water_content, kersten_type
    )
    
    # 7. Compilation des résultats
    results = {
        # Propriétés physiques
        'physical_properties': basic_props,
        'unfrozen_water_content': theta_u,
        'temperature': temperature,
        
        # Propriétés thermiques
        'heat_capacity_frozen': C_frozen,
        'heat_capacity_unfrozen': C_unfrozen,
        'latent_heat_fusion': L_soil,
        
        # Conductivités thermiques - Côté & Konrad
        'thermal_conductivity': {
            'solid_phase': lambda_s,
            'dry_soil': lambda_dry,
            'saturated_unfrozen': lambda_sat_u,
            'saturated_frozen': lambda_sat_f,
            'unfrozen_cote_konrad': lambda_u,
            'frozen_cote_konrad': lambda_f
        },
        
        # Vérification Kersten
        'thermal_conductivity_kersten': {
            'unfrozen_kersten': lambda_u_kersten,
            'frozen_kersten': lambda_f_kersten
        },
        
        # Paramètres d'entrée
        'input_parameters': {
            'density_dry': density_dry,
            'density_wet': density_wet,
            'water_content': water_content,
            'quartz_fraction': quartz_fraction,
            'soil_type': soil_type,
            'material_type': material_type,
            'specific_gravity': specific_gravity,
            'specific_surface': specific_surface
        }
    }
    
    return results

