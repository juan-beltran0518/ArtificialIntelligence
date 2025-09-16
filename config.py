"""
Configuracion Basica para Parcial IA
Autor: Juan Sebastián Beltrán
Fecha: Septiembre 15, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')

# Configuracion de visualizacion
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Configuracion de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Configuracion de seaborn
sns.set_palette("husl")

# Configuracion de numpy
np.random.seed(42)

print("✅ Configuracion basica cargada correctamente")