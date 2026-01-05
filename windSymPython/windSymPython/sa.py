import numpy as np
import scipy.io 
import matplotlib.pyplot as plt
import time 
import copy
import os
from f_powerPlantsT_fast import f_powerPlantsT_fast
import logging


# Parámetros iniciales
N_TURBINAS = 5     # o 50 turbinas
GRID_SIZE  = 5     # o 50 x 50
N_CELDAS   = GRID_SIZE * GRID_SIZE
MAX_EVALUACIONES = 3000 

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(f"windSymPython/windSymPython/log_{N_TURBINAS}.txt", mode="w", encoding="utf-8")]
)


def cargar_datos_viento(): 
    """
        Cargar los datos del viento desde el fichero MAT.

        Returns:
            numpy.ndarray | None: Vector `vVec` con las velocidades del viento (forma 2xn)
                si el fichero se carga correctamente, o `None` si no se encuentra el fichero.
    """

    try : 
        mat_data = scipy.io.loadmat('../windSym/dt/WindSym_1.mat')
        return mat_data["vVec"]
    
    except FileNotFoundError: 
        print("Fichero de datos no encontrado")
        return None
    

def generar_solucion_inicial(n_turbinas, grid_size): 
    """
        Generar una solución inicial aleatoria con `N_TURBINAS` ubicadas en el grid.
        Args:
            n_turbinas (int): Número de turbinas a colocar.
            grid_size (int): Tamaño del grid (GRID_SIZE x GRID_SIZE).
        Returns:
            numpy.ndarray: Matriz de forma `(GRID_SIZE, GRID_SIZE)` con valores 0/1 donde
                1 indica la presencia de una turbina.
    """

    # Generar un vector de ceros
    n_celdas = grid_size * grid_size
    vector = np.zeros(n_celdas, dtype=int)
    
    # Poner 1s en las primeras N_TURBINAS posiciones
    vector[:n_turbinas] = 1
    
    # Mezcla aleatoria
    np.random.shuffle(vector)
    
    # convertir a matriz 
    return vector.reshape((grid_size, grid_size))

def mover_turbina(solucion_actual): 
    """
        Mover aleatoriamente una turbina desde su posición actual a un hueco libre.

        Args:
            solucion_actual (numpy.ndarray): Matriz `(GRID_SIZE, GRID_SIZE)` con 0/1.

        Returns:
            numpy.ndarray: Nueva matriz con la turbina movida (misma forma que la entrada).
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

def dibujar_y_guardar_layout(layout, potencia, nombre_fichero="layout_final.png"):
    """
        Dibujar y guardar el layout final de turbinas.

        Args:
            layout (numpy.ndarray): Matriz `(GRID_SIZE, GRID_SIZE)` con 0/1.
            potencia (float): Potencia total del layout en MW.
            nombre_fichero (str): Nombre del fichero donde se guardará la imagen.
    """
    plt.figure(figsize=(8, 8))
    
    # Pintamos: 0 (vacío) en azul oscuro, 1 (turbina) en verde brillante
    plt.imshow(layout, cmap='ocean', interpolation='nearest') 
    
    # Conversión de unidades para el título (W -> MW)
    plt.title(f"Layout Final - Potencia: {potencia/1e6:.2f} MW")
    
    # Elementos auxiliares
    plt.colorbar(label="Presencia de Turbina (1=Sí, 0=No)")
    plt.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Guardar la figura y cerrarla
    plt.savefig(nombre_fichero)
    plt.close()

def sa(vVec): 
    """
        Algoritmo de Temple Simulado (Simulated Annealing) para optimizar la
        disposición de turbinas en el grid.

        Args:
            vVec (numpy.ndarray): Matriz o vector 2xn con las velocidades del viento.

        Returns:
            tuple:
                - mejor_solucion (numpy.ndarray): Matriz `(GRID_SIZE, GRID_SIZE)` con la
                  mejor disposición de turbinas encontrada (0/1).
                - mejor_potencia (float): Potencia asociada a `mejor_solucion` en MW.
                - historial_fitness (list[float]): Lista con la mejor potencia registrada
                  tras cada evaluación.
    """

    # Parámetros del algoritmo
    T_inicial = 10000.0
    T_final = 0.01

    MODO_ESTANCAMIENTO = True  # Activar criterio de estancamiento
    MAX_ESTANCAMIENTO = 200     # límite de estancamiento, 
                          # si pasa de este límite y no 
                          # hay mejor el algoritmo para. 
    contador_estancamiento = 0

    # Alpha para la actualización de la temperatura
    alpha = (T_final / T_inicial) ** (1.0 / MAX_EVALUACIONES)

    t_actual = T_inicial

    # Inicializar con una solución aleatoria
    solucion_actual = generar_solucion_inicial(N_TURBINAS, GRID_SIZE)

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
                contador_estancamiento = 0  # hay mejora, se reinicia el contador
                logging.info(f"Evaluación: {evaluaciones:2}, Nueva mejor potencia: {mejor_potencia:.2f} MW, Temperatura: {t_actual:.2f}")
        
        else :  # Delta negativo
            # Es peor solución, se acepta con una probabilidad
            contador_estancamiento += 1
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
        if MODO_ESTANCAMIENTO : 
            if contador_estancamiento >= MAX_ESTANCAMIENTO :
                print(" === PARADA POR ESTANCAMIENTO === ")
                print(f"Última mejor potencia: {mejor_potencia:.2f} MW")
                logging.info(f"=== PARADA POR ESTANCAMIENTO === Última mejor potencia: {mejor_potencia:.2f} MW")
                break
        
        if evaluaciones % 500 == 0 :
            logging.info(f"Evaluación: {evaluaciones/MAX_EVALUACIONES}| T={t_actual:.2f} | Actual={potencia_actual:.2f} | Mejor={mejor_potencia:.2f}")

    return mejor_solucion, mejor_potencia, historial_fitness


# Ejecución
if __name__ == "__main__" : 

    # Cargar datos del viento
    vVec = cargar_datos_viento()
    if vVec is None : 
        exit(1)

    # Ejecutar una sola vez 
    inicio = time.time()
    mejor_sol, mejor_pot, historial = sa(vVec)
    fin = time.time()

    print("\n=== Resultados Finales ===")
    print(f"Mejor potencia: {mejor_pot:.2f} MW")
    print(f"Mejor configuración de turbinas:\n{mejor_sol}")
    print(f"Tiempo de ejecución: {fin - inicio:.2f} segundos")

    # Graficar el historial de fitness
    plt.figure()
    plt.plot(historial)
    plt.title(f"Evolución de la Mejor Potencia con SA {GRID_SIZE} x {GRID_SIZE} \n(criterio parada por estancamiento activo)")
    plt.xlabel("Evaluaciones")
    plt.ylabel("Potencia (MW)")
    plt.grid(True)
    plt.savefig(f"fitness_evolution_{N_TURBINAS}turbinas.png")
    print(f"Gráfica guardada: fitness_evolution_{N_TURBINAS}turbinas.png")
    plt.show()

    # Guardar Mapa (Layout)
    dibujar_y_guardar_layout(mejor_sol, mejor_pot, f"layout_{N_TURBINAS}turbinas.png")
    print(f"Mapa guardado: layout_{N_TURBINAS}turbinas.png")
