Librairie Python pour le traitement des données expérimentales de modélisation physique de l'érosion du pergélisol côtier en canal à vagues.
Objectif : Automatiser l'analyse des données multi-instrumentales et standardiser les calculs hydrauliques/thermiques.

📦 Modules
🌊 1. wave_processing.py - Données de vagues

Analyse spectrale (méthode de Welch)
Décomposition onde incidente/réfléchie (Zelt & Skjelbreia 1992)
Paramètres : Hm0, Tp, Kr, spectres d'énergie
Source : Capteurs de hauteur d'eau (WG1-WG8)

🌡️ 2. rtd_processing.py - Données de température

Analyse du processus de dégel du pergélisol
Calcul des transferts de chaleur sensible/latente
Paramètres : vitesse de dégel, gradients dT/dx, flux de chaleur
Source : Capteurs RTD (RTD1-RTD8)

⚡ 3. adv_processing.py - Vélocimétrie ponctuelle

Analyse des vitesses d'écoulement et turbulence
Filtrage qualité (SNR, corrélation)
Paramètres : u*, TKE, contraintes de Reynolds
Source : ADV (Acoustic Doppler Velocimeter)

📊 4. profiler_processing.py - Profileur de vitesse

Profils verticaux de vitesse multi-cellules
Analyse de couche limite et cisaillement
Paramètres : u*, z₀, gradients dU/dz, profils logarithmiques
Source : Vectrino Profiler (.mat)

🧪 5. thermal_properties.py - Propriétés thermiques

Calcul propriétés thermiques sols gelés (Côté & Konrad 2005)
Validation par modèle Kersten (1949)
Paramètres : Cf, Cu, λf, λu, chaleur latente
Source : Données géotechniques

📋 Formats supportés

Vagues : CSV/Excel (time vector + élévations)
RTD : Excel (time vector + températures)
ADV : .dat + .hdr (binaire + header)
Profileur : .mat (Matlab)
Propriétés : Excel (données géotechniques)


📞Développeurs :
Omonigbehin, Olorunfemi Adeyemi
Hatim Ben Said

📞Encadrants :
Jacob Stolle


📚 Références

Zelt & Skjelbreia (1992) - Estimating Incident and Reflected Wave Fields Using an Arbitrary Number of Wave Gauges
Côté & Konrad (2005a) - Thermal conductivity of base-course materials
Côté & Konrad (2005b) - A generalized thermal conductivity model for soils and construction materials
