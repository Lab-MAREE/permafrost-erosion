# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 17:42:01 2025

@author: Hatim.BEN_SAID
"""

"""
Fonctions de traitement des données de vagues pour modélisation physique
Analyse spectrale, décomposition onde incidente/réfléchie, calcul des paramètres caractéristiques
Adapté pour l'étude de l'érosion du pergélisol côtier en laboratoire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy.signal import find_peaks
import scipy.integrate as integrate
import math

def get_wave_spectra_psd(t, eta):
    '''
    Analyse spectrale de densité de puissance des vagues par FFT brute (sans fenêtrage)
    
    Intrants:
    - t : numpy array - Vecteur temps pour l'élévation de surface (s)
    - eta : numpy array - Vecteur d'élévation de surface d'eau (m)
    
    Extrants:
    - f : numpy array - Vecteur fréquence (Hz)
    - S : numpy array - Densité spectrale de puissance (m²/Hz)
    - a : numpy array - Spectre d'amplitude (m)
    
    Format des intrants:
    - t et eta doivent avoir la même longueur
    - Échantillonnage régulier recommandé
    '''
    y = eta - np.mean(eta)
    sample_rate = np.mean(np.gradient(t))
    N = round((np.max(t) - np.min(t))/sample_rate)
    yf = rfft(y)
    f = rfftfreq(N, sample_rate)
    if len(yf) != len(f):
        yf = yf[:-1]
    a = 2*abs(yf/N)
    S = np.square(a)/(2*np.mean(np.gradient(f)))
    return f, S, a

def get_wave_spectra_welch(t, eta):
    '''
    Analyse spectrale des vagues par la méthode de Welch avec fenêtre de Hanning
    Recommandée pour les signaux longs avec du bruit
    
    Intrants:
    - t : numpy array - Vecteur temps pour l'élévation de surface (s)
    - eta : numpy array - Vecteur d'élévation de surface d'eau (m)
    
    Extrants:
    - f : numpy array - Vecteur fréquence (Hz)
    - S : numpy array - Densité spectrale de puissance (m²/Hz)
    
    Format des intrants:
    - t et eta doivent avoir la même longueur
    - Durée minimale recommandée: 300-600 secondes pour bonnes statistiques
    '''
    y = eta - np.mean(eta)
    Fs = 1/np.mean(np.gradient(t[0:100]))
    f, S = signal.welch(y, fs=Fs, window='hann', nperseg=1024)
    return f, S

def calculate_wave_characteristics(f, S):
    '''
    Calcul des paramètres caractéristiques des vagues à partir du spectre
    
    Intrants:
    - f : numpy array - Vecteur fréquence (Hz)
    - S : numpy array - Densité spectrale de puissance (m²/Hz)
    
    Extrants:
    - Hm0 : float - Hauteur significative spectrale (m)
    - Tp : float - Période de pic spectral (s)
    
    Format des intrants:
    - f et S doivent avoir la même longueur
    - f doit être croissant et commencer proche de 0
    '''
    S = np.nan_to_num(S)
    m0 = integrate.trapezoid(S, f)  # Moment d'ordre 0
    Hm0 = 4 * np.sqrt(m0)  # Hauteur significative
    max_S_index = np.argmax(S)
    if f[max_S_index] != 0:
        Tp = 1 / f[max_S_index]  # Période de pic
    else:
        Tp = np.nan
    return Hm0, Tp

def calculate_Hrms(t, eta):
    '''
    Calcul de Hrms et période moyenne par analyse temporelle (méthode zero-crossing)
    
    Intrants:
    - t : numpy array - Vecteur temps (s)
    - eta : numpy array - Élévation de surface d'eau (m)
    
    Extrants:
    - Hrms : float - Hauteur RMS des vagues (m)
    - Tm : float - Période moyenne (s)
    
    Format des intrants:
    - Signaux échantillonnés à fréquence constante
    - Durée suffisante pour capturer plusieurs vagues complètes
    '''
    eta = eta - eta[0:min(15000, len(eta))].mean()
    zero_crossings = np.where(np.diff(np.sign(eta)))[0]
    zero_crossings = zero_crossings[::2]  # Garde un passage sur deux
    
    T = np.full(len(zero_crossings)-1, np.nan)
    H = np.full(len(zero_crossings)-1, np.nan)
    
    for j in range(0, len(zero_crossings)-1):
        T[j] = t[zero_crossings[j+1]] - t[zero_crossings[j]]
        H[j] = eta[zero_crossings[j]:zero_crossings[j+1]].max() - eta[zero_crossings[j]:zero_crossings[j+1]].min()
    
    Hrms = np.sqrt(np.power(H, 2).sum()/len(H))
    Tm = T.mean()
    return Hrms, Tm

def linear_wave_length(T, h, g=9.81):
    '''
    Calcul de la longueur d'onde linéaire par résolution itérative de la relation de dispersion
    
    Intrants:
    - T : float - Période de la vague (s)
    - h : float - Profondeur d'eau (m)
    - g : float - Accélération gravitationnelle (m/s²), défaut = 9.81
    
    Extrants:
    - L : float - Longueur d'onde linéaire (m)
    
    Format des intrants:
    - T > 0, h > 0
    - Valable pour toute profondeur (eau peu profonde à profonde)
    '''
    if T <= 1e-6:
        return np.nan
    
    # Approximation initiale
    L2 = g * T**2 / (2 * np.pi) * np.sqrt(np.tanh(4 * np.pi**2 * h / (T**2 * g)))
    if np.isnan(L2) or L2 <= 0:
        return np.nan
    
    # Résolution itérative
    try:
        L0 = np.arange(L2 * 0.90, L2 * 1.10, 0.01)
    except ValueError:
        return np.nan
    
    temp = [abs(L - g * T**2 / (2 * np.pi) * np.tanh(2 * np.pi * h / L)) for L in L0]
    indL = np.argmin(temp)
    return L0[indL]

def jonswap_spectrum(f, Hm0, Tp, gamma=3.3):
    '''
    Calcul du spectre JONSWAP théorique normalisé pour correspondre au Hm0 mesuré
    
    Intrants:
    - f : numpy array - Vecteur des fréquences (Hz)
    - Hm0 : float - Hauteur significative mesurée (m)
    - Tp : float - Période de pic (s)
    - gamma : float - Facteur d'amélioration du pic (défaut: 3.3)
    
    Extrants:
    - S_jonswap : numpy array - Densité spectrale JONSWAP (m²/Hz)
    
    Format des intrants:
    - f doit être un vecteur croissant commençant près de 0
    - Hm0 > 0, Tp > 0
    '''
    g = 9.81
    fp = 1.0 / Tp  # fréquence de pic
    
    # Éviter les fréquences nulles
    f_safe = np.where(f == 0, 1e-10, f)
    
    # Facteur sigma pour la forme asymétrique
    sigma = np.where(f_safe <= fp, 0.07, 0.09)
    
    # Terme d'amélioration du pic
    r = np.exp(-0.5 * ((f_safe - fp) / (sigma * fp))**2)
    enhancement = gamma**r
    
    # Spectre JONSWAP normalisé
    S_normalized = (g**2 / ((2*np.pi)**4 * f_safe**5) * 
                   np.exp(-5/4 * (fp/f_safe)**4) * enhancement)
    
    # Normalisation pour obtenir le bon Hm0
    m0_normalized = np.trapz(S_normalized, f)
    m0_desired = (Hm0 / 4.0)**2
    
    if m0_normalized > 0:
        alpha = m0_desired / m0_normalized
    else:
        alpha = 0.0081
    
    S_jonswap = alpha * S_normalized
    return S_jonswap

def butter_highpass_filter(data, cutoff, fs, order=5):
    '''
    Filtrage passe-haut Butterworth pour éliminer les basses fréquences parasites
    
    Intrants:
    - data : numpy array - Données à filtrer
    - cutoff : float - Fréquence de coupure (Hz)
    - fs : float - Fréquence d'échantillonnage (Hz)
    - order : int - Ordre du filtre (défaut: 5)
    
    Extrants:
    - y : numpy array - Données filtrées
    
    Format des intrants:
    - cutoff < fs/2 (critère de Nyquist)
    - Recommandé: cutoff entre 0.01-0.05 Hz pour éliminer la dérive
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def zeltAndSkelbreia1992(data, t, h, pos, Fs):
    '''
    Décomposition onde incidente/réfléchie selon la méthode de Zelt & Skjelbreia (1992)
    Utilise un réseau de capteurs pour séparer les composantes directionnelles
    
    Intrants:
    - data : numpy array (N x M) - Données des M capteurs sur N points temporels
    - t : numpy array - Vecteur temps (s)
    - h : float - Profondeur d'eau (m)
    - pos : numpy array - Positions des capteurs le long du canal (m)
    - Fs : float - Fréquence d'échantillonnage (Hz)
    
    Extrants:
    - N : numpy array - Série temporelle totale reconstruite (m)
    - Ni : numpy array - Série temporelle onde incidente (m)
    - Nr : numpy array - Série temporelle onde réfléchie (m)
    - Ai : numpy array - Amplitudes incidentes par fréquence
    - Ar : numpy array - Amplitudes réfléchies par fréquence
    
    Format des intrants:
    - Minimum 3 capteurs, idéalement 5-8
    - Espacement entre capteurs: L/10 à L/4 (L = longueur d'onde)
    - Données synchronisées et de même durée
    '''
    g = 9.81
    dx = pos - pos[0]  # Distance depuis le premier capteur
    weights = 1  # Poids uniforme (peut être modifié selon la qualité des capteurs)
    
    _, c = data.shape
    nfft = len(data)
    Y = np.empty((len(data)//2 + 1, c), dtype=np.complex128)
    
    # Transformée de Fourier pour chaque capteur
    for i in range(c):
        Y[:, i] = rfft(data[:, i])/nfft

    fint = rfftfreq(len(data), 1/Fs)
    om = 2 * np.pi * fint

    Ai = np.zeros(len(fint))
    Ar = np.zeros(len(fint))
    N = np.zeros(len(t))
    Ni = np.zeros(len(t))
    Nr = np.zeros(len(t))

    # Décomposition pour chaque fréquence
    for i in range(len(fint)):
        if fint[i] < 0.001 or fint[i] > 10:  # Limite de fréquences valides
            continue

        L = linear_wave_length(1/fint[i], h, g)
        k = 2 * np.pi / L

        # Coefficients de propagation incidente et réfléchie
        Ci = np.exp(-1j * k * dx) / 2
        Cr = np.exp(1j * k * dx) / 2

        # Système d'équations pour la décomposition
        A11 = np.sum(weights * (Ci**2))
        A12 = np.sum(weights * (Ci * Cr))
        A22 = np.sum(weights * (Cr**2))

        B1 = np.sum(weights * Y[i, :] * Ci)
        B2 = np.sum(weights * Y[i, :] * Cr)

        A = np.array([[A11, A12], [A12, A22]])
        B = np.array([B1, B2])
        
        # Résolution du système
        result = np.linalg.lstsq(A, B, rcond=None)
        X = result[0].flatten()

        ai, ar = np.abs(X[0]), np.abs(X[1])
        phi, phr = -np.angle(X[0])-k*pos[0], np.angle(X[1])-k*pos[0]

        Ai[i], Ar[i] = ai, ar
        
        # Reconstruction des séries temporelles
        N += ai*np.cos(k*pos[0]-om[i]*t+phi) + ar*np.cos(k*pos[0]+om[i]*t+phr)
        Ni += ai*np.cos(k*pos[0]-om[i]*t+phi)
        Nr += ar*np.cos(k*pos[0]+om[i]*t+phr)

    return N, Ni, Nr, Ai, Ar

def calculate_reflection_coefficient(Si, Sr, fi, fr, cut_off_freq=3.0):
    '''
    Calcul du coefficient de réflexion énergétique Kr à partir des spectres
    
    Intrants:
    - Si : numpy array - Spectre incident (m²/Hz)
    - Sr : numpy array - Spectre réfléchi (m²/Hz)
    - fi : numpy array - Fréquences du spectre incident (Hz)
    - fr : numpy array - Fréquences du spectre réfléchi (Hz)
    - cut_off_freq : float - Fréquence de coupure pour l'intégration (Hz)
    
    Extrants:
    - Kr : float - Coefficient de réflexion énergétique (sans dimension)
    - energy_incident : float - Énergie incidente (J/m²)
    - energy_reflected : float - Énergie réfléchie (J/m²)
    
    Format des intrants:
    - Spectres provenant de get_wave_spectra_welch()
    - Même résolution fréquentielle recommandée
    '''
    rho = 1000  # Densité de l'eau (kg/m³)
    g = 9.81    # Accélération gravitationnelle (m/s²)
    
    # Indices de coupure
    cut_off_index_i = np.argmax(fi >= cut_off_freq)
    cut_off_index_r = np.argmax(fr >= cut_off_freq)
    
    # Calcul des énergies
    energy_incident = rho * g * np.trapz(Si[:cut_off_index_i], fi[:cut_off_index_i])
    energy_reflected = rho * g * np.trapz(Sr[:cut_off_index_r], fr[:cut_off_index_r])
    
    # Coefficient de réflexion
    Kr = np.sqrt(energy_reflected / energy_incident) if energy_incident > 0 else 0
    
    return Kr, energy_incident, energy_reflected

def compare_with_jonswap(f, S_measured, Hm0, Tp):
    '''
    Compare un spectre mesuré avec le spectre JONSWAP théorique
    
    Intrants:
    - f : numpy array - Vecteur fréquence (Hz)
    - S_measured : numpy array - Spectre mesuré (m²/Hz)
    - Hm0 : float - Hauteur significative (m)
    - Tp : float - Période de pic (s)
    
    Extrants:
    - stats : dict - Dictionnaire contenant les statistiques de comparaison
        - correlation : coefficient de corrélation
        - rmse_normalized : erreur quadratique moyenne normalisée
        - peak_error_percent : erreur relative sur les pics (%)
    
    Format des intrants:
    - Spectre mesuré depuis get_wave_spectra_welch()
    - Paramètres Hm0, Tp depuis calculate_wave_characteristics()
    '''
    # Calcul du spectre JONSWAP
    S_jonswap = jonswap_spectrum(f, Hm0, Tp)
    
    # Statistiques de comparaison
    correlation = np.corrcoef(S_measured, S_jonswap)[0, 1]
    rmse = np.sqrt(np.mean((S_measured - S_jonswap)**2)) / np.mean(S_measured)
    peak_error = abs(np.max(S_measured) - np.max(S_jonswap)) / np.max(S_measured) * 100
    
    stats = {
        'correlation': correlation,
        'rmse_normalized': rmse,
        'peak_error_percent': peak_error,
        'S_jonswap': S_jonswap
    }
    
    return stats

def preprocess_wave_data(data_df, start_time, duration_minutes):
    '''
    Préprocessing des données de vagues: découpage temporel et nettoyage
    
    Intrants:
    - data_df : pandas DataFrame - Données brutes avec index temporel
    - start_time : str ou pd.Timestamp - Temps de début d'analyse (format: "YYYY-MM-DD HH:MM:SS")
    - duration_minutes : float - Durée d'analyse en minutes
    
    Extrants:
    - data_trimmed : pandas DataFrame - Données découpées et nettoyées
    
    Format des intrants:
    - DataFrame avec index temporel (pd.DatetimeIndex)
    - Colonnes: capteurs de vagues (ex: 'WG1', 'WG2', etc.)
    '''
    start_time = pd.to_datetime(start_time)
    end_time = start_time + pd.Timedelta(minutes=duration_minutes)
    
    # Découpage temporel
    data_trimmed = data_df.loc[start_time:end_time]
    
    # Interpolation linéaire pour les valeurs manquantes
    data_trimmed = data_trimmed.interpolate(method='linear', limit_direction='both', axis=0)
    
    return data_trimmed