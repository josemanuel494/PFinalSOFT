import numpy as np
import scipy.io 
import matplotlib.pyplot as plt
import time 
import copy
import os
from f_powerPlantsT_fast import f_powerPlantsT_fast
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("windSymPython/windSymPython/log.txt", mode="w", encoding="utf-8")]
)

# Parámetros iniciales
N_TURBINAS = 20     # o 50 turbinas
GRID_SIZE  = 20     # o 50 x 50
N_CELDAS   = GRID_SIZE * GRID_SIZE
MAX_EVALUACIONES = 3000 


def cargar_datos_viento(): 
    """
        Función para cargar los datos del viento proporcionados
    """

    try : 
        mat_data = scipy.io.loadmat('../windSym/dt/WindSym_1.mat')
        return mat_data["vVec"]
    
    except FileNotFoundError: 
        print("Fichero de datos no encontrado")
        return None
    
def generar_solucion_inicial(): 
    """
        Función para generar un grid aleatorio con N_TURBINAS
    """

    # Generar un vector de ceros
    vector = np.zeros(N_CELDAS, dtype=int)
    
    # Poner 1s en las primeras N_TURBINAS posiciones
    vector[:N_TURBINAS] = 1
    
    # Mezcla aleatoria
    np.random.shuffle(vector)
    
    # convertir a matriz 
    return vector.reshape((GRID_SIZE, GRID_SIZE))

def mover_turbina(solucion_actual): 
    """
        Función para mover una turbina de una 
        posición (1) a otra posición (0). 
    """

    vecino = np.copy(solucion_actual)

    # Obtener las coordenadas de las turbinas (1s)
    coords_turbinas = np.argwhere(vecino == 1)

    # Obtener las coordenadas de los huecos (0s)
    coords_huecos = np.argwhere(vecino == 0)

    # Elección aleatoria de una turbina y un hueco
    idx_turbina = coords_turbinas[np.random.randint(len(coords_turbinas))]
    idx_hueco = coords_huecos[np.random.randint(len(coords_huecos))]
    
    # Mover la turbina del lugar actual al nuevo lugar
    vecino[idx_turbina[0], idx_turbina[1]] = 0
    vecino[idx_hueco[0], idx_hueco[1]] = 1

    return vecino

def sa(vVec): 
    """
        Algoritmo de Temple Simulado (Simulated Annealing)
        Params:     
            vVec: vector de 2xn veloidades del viento

        Returns:
            mejor_solucion: mejor solución encontrada
            mejor_potencia: potencia de la mejor solución
            historial_fitness: historial de potencias durante la ejecución
    """

    # Parámetros del algoritmo
    T_inicial = 10000.0
    T_final = 0.01

    # MAX_ESTANCAMIENTO = 100     # límite de estancamiento, 
                            # si pasa de este límite y no 
                            # hay mejor el algoritmo para. 
    # contador_estancamiento = 0

    # Alpha para la actualización de la temperatura
    alpha = (T_final / T_inicial) ** (1.0 / MAX_EVALUACIONES)

    t_actual = T_inicial

    # Inicializar con una solución aleatoria
    solucion_actual = generar_solucion_inicial()

    # Evaluar la solución inicial
    resultados = f_powerPlantsT_fast(vVec, solucion_actual)
    potencia_actual = resultados[0]

    # Mejor solución encontrada
    mejor_solucion = np.copy(solucion_actual)
    mejor_potencia = potencia_actual

    # Guardar número de evaluaciones y potencias para graficar
    evaluaciones = 1
    historial_fitness = [potencia_actual]

    logging.info(f"Inicio SA. Potencias inicial {potencia_actual:.2f} MW")

    # Bucle principal
    while evaluaciones < MAX_EVALUACIONES : 

        # Generar un vecino
        solucion_vecina = mover_turbina(solucion_actual)

        # Evaluar el vecino 
        resultados_vecino = f_powerPlantsT_fast(vVec, solucion_vecina)
        potencia_vecina = resultados_vecino[0]
        evaluaciones += 1

        # Calcular la diferencia de potencia
        delta = potencia_vecina - potencia_actual

        # Decidir si se acepta el vecino
        aceptar = False
        if delta > 0 : 
            # Es mejor solución
            aceptar = True
            
            # Comprobar si es la mejor solución
            if potencia_vecina > mejor_potencia : 
                mejor_potencia = potencia_vecina
                mejor_solucion = np.copy(solucion_vecina)
                # contador_estancamiento = 0  # hay mejora, se reinicia el contador
                logging.info(f"Evaluación: {evaluaciones:2}, Nueva mejor potencia: {mejor_potencia:.2f} MW, Temperatura: {t_actual:.2f}")
        
        else :  # Delta negativo
            # Es peor solución, se acepta con una probabilidad
            # contador_estancamiento += 1
            probabilidad = np.exp(delta / t_actual)

            if np.random.rand() < probabilidad :
                aceptar = True

        # Si se acepta, actualizar la solución actual
        if aceptar :
            solucion_actual = solucion_vecina
            potencia_actual = potencia_vecina

        # Actualizar la temperatura
        t_actual *= alpha
        historial_fitness.append(mejor_potencia)

        # Comprobar el criterio de estancamiento
        # if contador_estancamiento >= MAX_ESTANCAMIENTO :
        #     print(" === PARA POR ESTANCAMIENTO === ")
        #     print(f"Última mejor potencia: {mejor_potencia:.2f} MW")
        #     logging.info(f"=== PARA POR ESTANCAMIENTO === Última mejor potencia: {mejor_potencia:.2f} MW")
        #     break
        
        if evaluaciones % 500 == 0 :
            logging.info(f"Evaluación: {evaluaciones/MAX_EVALUACIONES}| T={t_actual:.2f} | Actual={potencia_actual:.2f} | Mejor={mejor_potencia:.2f}")

    return mejor_solucion, mejor_potencia, historial_fitness


# Ejecución
if __name__ == "__main__" : 

    # Cargar datos del viento
    vVec = cargar_datos_viento()
    if vVec is None : 
        exit(1)

    # Ejecutar SA
    inicio = time.time()
    mejor_sol, mejor_pot, historial = sa(vVec)
    fin = time.time()

    print("\n=== Resultados Finales ===")
    print(f"Mejor potencia: {mejor_pot:.2f} MW")
    print(f"Mejor configuración de turbinas:\n{mejor_sol}")
    print(f"Tiempo de ejecución: {fin - inicio:.2f} segundos")

    # Graficar el historial de fitness
    plt.plot(historial)
    plt.title("Evolución de la Mejor Potencia con SA")
    plt.xlabel("Número de Evaluaciones")
    plt.ylabel("Mejor Potencia (MW)")
    plt.grid(True)
    plt.show()