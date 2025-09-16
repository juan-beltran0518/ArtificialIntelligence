"""
Utilidades Basicas - Parcial IA
Funciones esenciales para analisis de datos

Autor: Juan Sebastián Beltrán
Fecha: Septiembre 15, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_inspect_data(file_path, show_info=True):
    """
    Carga y realiza inspeccion basica de un dataset
    
    Parameters:
    -----------
    file_path : str
        Ruta al archivo CSV
    show_info : bool
        Mostrar informacion del dataset
    
    Returns:
    --------
    pd.DataFrame
        Dataset cargado
    """
    try:
        df = pd.read_csv(file_path)
        
        if show_info:
            print("INFORMACION DEL DATASET")
            print("="*40)
            print(f"Dimensiones: {df.shape}")
            print(f"Columnas: {list(df.columns)}")
            print(f"\nPrimeras 5 filas:")
            print(df.head())
            
        return df
        
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

def check_missing_values(df, visualize=True):
    """
    Analiza valores faltantes en el dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    visualize : bool
        Crear visualizacion
    
    Returns:
    --------
    pd.DataFrame
        Resumen de valores faltantes
    """
    missing_data = pd.DataFrame({
        'Columna': df.columns,
        'Valores_Faltantes': df.isnull().sum(),
        'Porcentaje': (df.isnull().sum() / len(df)) * 100
    })
    
    missing_data = missing_data[missing_data['Valores_Faltantes'] > 0]
    
    print("VALORES FALTANTES")
    print("="*30)
    
    if missing_data.empty:
        print("✅ No hay valores faltantes")
    else:
        print(missing_data)
        
        if visualize:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=missing_data, x='Porcentaje', y='Columna')
            plt.title('Valores Faltantes por Columna')
            plt.show()
    
    return missing_data

def detect_outliers(df, method='iqr'):
    """
    Detecta outliers en variables numericas
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    method : str
        Metodo de deteccion ('iqr' o 'zscore')
    
    Returns:
    --------
    dict
        Diccionario con outliers por columna
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outliers_dict = {}
    
    print(f"DETECCION DE OUTLIERS")
    print("="*30)
    
    for col in numerical_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[z_scores > 3]
        
        outliers_dict[col] = outliers
        
        print(f"{col}: {len(outliers)} outliers ({(len(outliers)/len(df)*100):.1f}%)")
    
    return outliers_dict

def correlation_analysis(df, threshold=0.5):
    """
    Analisis basico de correlacion
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    threshold : float
        Umbral para correlaciones fuertes
    """
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.empty:
        print("❌ No hay variables numericas")
        return None
    
    print("MATRIZ DE CORRELACION")
    print("="*30)
    
    corr_matrix = numerical_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlacion')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

print("✅ Utilidades basicas cargadas correctamente")