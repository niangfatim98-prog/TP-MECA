import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Données SANS pesanteur
F_mes_disque = np.array([0.540, 1.485, 2.044, 2.700, 3.529, 3.780])
F_Euler_disque = np.array([0.861, 1.744, 2.355, 3.021, 3.852, 4.436])

F_mes_hemi = np.array([0.347, 1.659, 2.469, 4.089, 5.943])
F_Euler_hemi = np.array([0.550, 1.912, 2.926, 4.622, 7.046])

# Données AVEC pesanteur (valeurs théoriques avec gravité)
F_Euler_grav_disque = np.array([1.672, 2.543, 3.150, 3.812, 4.643, 5.227])
F_Euler_grav_hemi = np.array([2.599, 3.961, 4.975, 6.671, 9.095])

# --- Figure ---
plt.figure(figsize=(12, 5))

# 1) SANS pesanteur — régression sur tous les points combinés
plt.subplot(1, 2, 1)
F_mes_all = np.concatenate([F_mes_disque, F_mes_hemi])
F_Euler_all = np.concatenate([F_Euler_disque, F_Euler_hemi])

slope, intercept, r_value, p_value, std_err = stats.linregress(F_mes_all, F_Euler_all)
a_sans, b_sans, R2_sans = slope, intercept, r_value**2

x_fit = np.linspace(0, max(F_mes_all.max(), F_Euler_all.max(), 6), 200)
y_fit = a_sans * x_fit + b_sans

plt.plot(F_mes_disque, F_Euler_disque, 'bo', label='Disque plat', markersize=8)
plt.plot(F_mes_hemi, F_Euler_hemi, 'ro', label='Hémisphère', markersize=8)
plt.plot(x_fit, x_fit, 'k--', label='y=x (accord parfait)', linewidth=2)
plt.plot(x_fit, y_fit, 'g-', label=f'Régression: y={a_sans:.3f}x+{b_sans:.3f}\nR²={R2_sans:.3f}', linewidth=2)

plt.xlabel('F_mes (N) - Expérience')
plt.ylabel('F_Euler (N) - Théorie')
plt.title('SANS pesanteur')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')

# 2) AVEC pesanteur — utiliser les mêmes F_mes (mesurées), comparer aux F_Euler_grav
plt.subplot(1, 2, 2)
# On suppose que les mesures expérimentales (F_mes_disque/hemi) correspondent
# aux valeurs théoriques avec gravité fournies (mêmes longueurs respectives).
F_mes_grav_all = np.concatenate([F_mes_disque, F_mes_hemi])
F_Euler_grav_all = np.concatenate([F_Euler_grav_disque, F_Euler_grav_hemi])

# Vérifier compatibilité des tailles
if F_mes_grav_all.shape != F_Euler_grav_all.shape:
    raise ValueError('Les vecteurs F_mes et F_Euler_grav n\'ont pas la même longueur.')

slope_g, intercept_g, r_value_g, p_value_g, std_err_g = stats.linregress(F_mes_grav_all, F_Euler_grav_all)
a_grav, b_grav, R2_grav = slope_g, intercept_g, r_value_g**2

x_fit_g = np.linspace(0, max(F_mes_grav_all.max(), F_Euler_grav_all.max(), 6), 200)
y_fit_g = a_grav * x_fit_g + b_grav

# Tracé par jeu de données pour garder légendes séparées
n_disque = len(F_mes_disque)
plt.plot(F_mes_disque, F_Euler_grav_disque, 'bo', label='Disque plat', markersize=8)
plt.plot(F_mes_hemi, F_Euler_grav_hemi, 'ro', label='Hémisphère', markersize=8)
plt.plot(x_fit_g, x_fit_g, 'k--', label='y=x (accord parfait)', linewidth=2)
plt.plot(x_fit_g, y_fit_g, 'g-', label=f'Régression: y={a_grav:.3f}x+{b_grav:.3f}\nR²={R2_grav:.3f}', linewidth=2)

plt.xlabel('F_mes (N) - Expérience')
plt.ylabel('F_Euler avec pesanteur (N)')
plt.title('AVEC pesanteur')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')

plt.tight_layout()

# Sauvegarder la figure et l'afficher si possible
out_file = 'forces_plot.png'
plt.savefig(out_file, dpi=200)
try:
    plt.show()
except Exception:
    # En environnement sans affichage, plt.show() peut lever une exception
    pass

print(f"SANS pesanteur: F_Euler = {a_sans:.3f}·F_mes + {b_sans:.3f}, R² = {R2_sans:.3f}")
print(f"AVEC pesanteur: F_Euler_grav = {a_grav:.3f}·F_mes + {b_grav:.3f}, R² = {R2_grav:.3f}")
print(f"Graphique sauvegardé dans: {out_file}")
