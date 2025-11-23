import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Données SANS pesanteur
F_mes_disque = np.array([0.540, 1.485, 2.044, 2.700, 3.529, 3.780])
F_Euler_disque = np.array([0.861, 1.744, 2.355, 3.021, 3.852, 4.436])

F_mes_hemi = np.array([0.347, 1.659, 2.469, 4.089, 5.943])
F_Euler_hemi = np.array([0.550, 1.912, 2.926, 4.622, 7.046])

# Données AVEC pesanteur
F_Euler_grav_disque = np.array([1.672, 2.543, 3.150, 3.812, 4.643, 5.227])
F_Euler_grav_hemi = np.array([2.599, 3.961, 4.975, 6.671, 9.095])

# Graphique SANS pesanteur
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Tous points confondus
F_mes_all = np.concatenate([F_mes_disque, F_mes_hemi])
F_Euler_all = np.concatenate([F_Euler_disque, F_Euler_hemi])

# Régression linéaire : F_Euler = a * F_mes + b
slope, intercept, r_value, p_value, std_err = stats.linregress(F_mes_all, F_Euler_all)
a_sans, b_sans, R2_sans = slope, intercept, r_value**2

# Courbe
x_fit = np.linspace(0, 6, 100)
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

# Graphique AVEC pesanteur
plt.subplot(1, 2, 2)
# Graphique AVEC pesanteur
plt.subplot(1, 2, 2)

# Create masks to filter out invalid (zero or negative) values
mask_disque = (F_mes_disque > 0) & (F_Euler_grav_disque > 0)
mask_hemi = (F_mes_hemi > 0) & (F_Euler_grav_hemi > 0)

F_mes_valides = np.concatenate([F_mes_disque[mask_disque], F_mes_hemi[mask_hemi]])
F_Euler_grav_valides = np.concatenate([F_Euler_grav_disque[mask_disque], F_Euler_grav_hemi[mask_hemi]])

# Régression linéaire
# ... rest of your code

F_mes_valides = np.concatenate([F_mes_disque[mask_disque], F_mes_hemi[mask_hemi]])
F_Euler_grav_valides = np.concatenate([F_Euler_grav_disque[mask_disque], F_Euler_grav_hemi[mask_hemi]])

# Régression linéaire
if len(F_mes_valides) > 1:
  slope_grav, intercept_grav, r_value_grav, p_value_grav, std_err_grav = stats.linregress(F_mes_valides, F_Euler_grav_valides)
  a_grav, b_grav, R2_grav = slope_grav, intercept_grav, r_value_grav**2
else:
  a_grav, b_grav, R2_grav = 1, 0, 0

x_fit_grav = np.linspace(0, 6, 100)
y_fit_grav = a_grav * x_fit_grav + b_grav

plt.plot(F_mes_disque[mask_disque], F_Euler_grav_disque[mask_disque], 'bo', label='Disque plat', markersize=8)
plt.plot(F_mes_hemi[mask_hemi], F_Euler_grav_hemi[mask_hemi], 'ro', label='Hémisphère', markersize=8)
plt.plot(x_fit_grav, x_fit_grav, 'k--', label='y=x (accord parfait)', linewidth=2)
plt.plot(x_fit_grav, y_fit_grav, 'g-', label=f'Régression: y={a_grav:.3f}x+{b_grav:.3f}\nR²={R2_grav:.3f}', linewidth=2)

plt.xlabel('F_mes (N) - Expérience')
plt.ylabel('F_Euler avec pesanteur (N)')
plt.title('AVEC pesanteur')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')

plt.tight_layout()
out_file = 'tpcourantget/forces_plot.png'
plt.savefig(out_file, dpi=200)
try:
  plt.show()
except Exception:
  pass

print(f"SANS pesanteur: F_Euler = {a_sans:.3f}·F_mes + {b_sans:.3f}, R² = {R2_sans:.3f}")
print(f"AVEC pesanteur: F_Euler_grav = {a_grav:.3f}·F_mes + {b_grav:.3f}, R² = {R2_grav:.3f}")
print(f"Graphique sauvegardé dans: {out_file}")
