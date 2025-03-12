import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Analyzer:
    """
    Clase para analizar y visualizar la evolución de los valores de estado
    durante la iteración de políticas.
    """

    def __init__(self, mdp):
        """
        :param mdp: The MDP that is being analyzed
        """
        self.mdp = mdp
        self.runs = {}  # Diccionario para almacenar los datos de cada ejecución
        self.current_run = None
    
    def new_run(self, name):
        """
        :param name: The name of the run, used for figures
        
        Indica al analizador que las próximas observaciones son para una nueva ejecución
        del algoritmo de aprendizaje de políticas
        """
        self.runs[name] = {
            'iterations': [],
            'state_values': [],
            'iteration_count': 0
        }
        self.current_run = name
    
    def add_state_value_estimates(self, v):
        """
        :param v: dictionary with state values or estimates thereof
        
        Añade las estimaciones de valores de estado al historial de la ejecución actual
        """
        if self.current_run is None:
            raise ValueError("No current run set. Call new_run() first.")
        
        iteration = self.runs[self.current_run]['iteration_count']
        self.runs[self.current_run]['iterations'].append(iteration)
        self.runs[self.current_run]['state_values'].append(v.copy())
        self.runs[self.current_run]['iteration_count'] += 1
    
    def plot_state_value_estimates_of_init_state_over_time(self, ax=None):
        """
        :param ax: optional axis object where the lines are drawn. If not given, a new ax object is created

        Crea un gráfico de líneas (función escalonada) que muestra una línea para cada ejecución.
        El valor de la línea corresponde al valor del estado inicial del MDP.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        init_state = self.mdp.init_states[0]  # Tomamos el primer estado inicial
        
        for run_name, run_data in self.runs.items():
            iterations = run_data['iterations']
            values = [state_values[init_state] for state_values in run_data['state_values']]
            
            ax.step(iterations, values, where='post', label=run_name)
        
        ax.set_xlabel('Iteration t')
        ax.set_ylabel(r'$v_t(s_0)$')
        ax.legend()
        ax.grid(True)
        
        return ax

    def plot_avg_state_value_estimates_over_time(self, ax=None):
        """
        :param ax: optional axis object where the lines are drawn. If not given, a new ax object is created

        Crea un gráfico de líneas (función escalonada) que muestra una línea para cada ejecución.
        El valor de la línea corresponde al valor promedio de todos los estados en el MDP.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for run_name, run_data in self.runs.items():
            iterations = run_data['iterations']
            
            # Calcular el valor promedio de todos los estados en cada iteración
            avg_values = []
            for state_values in run_data['state_values']:
                # Filtramos estados terminales o estados con valores extremos
                valid_states = [s for s in state_values.keys() if not self.mdp.is_terminal_state(s)]
                if valid_states:
                    avg_value = sum(state_values[s] for s in valid_states) / len(valid_states)
                else:
                    avg_value = 0
                avg_values.append(avg_value)
            
            ax.step(iterations, avg_values, where='post', label=run_name)
        
        ax.set_xlabel('Iteration t')
        ax.set_ylabel(r'$\overline{v}_t(s)$')
        ax.legend()
        ax.grid(True)
        
        return ax
    
    def create_heatmap_of_state_values(self, v, title=None, ax=None, vmin=None, vmax=None):

        """
        Crea una visualización de mapa de calor de los valores de estado para un MDP basado en cuadrícula como Lake
    
        :param v: dictionary with state values
        :param title: optional title for the heatmap
        :param ax: optional axis object where the heatmap is drawn
        :param vmin: valor mínimo para la escala de color (opcional)
        :param vmax: valor máximo para la escala de color (opcional)
        :return: el objeto axis donde se dibujó el heatmap
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determinar el tamaño de la cuadrícula a partir del MDP del lago
        grid_shape = self.mdp.world.shape
        
        # Crear una matriz para el mapa de calor
        heatmap_data = np.zeros(grid_shape)
        
        # Llenar la matriz con los valores de los estados
        for s in v:
            if isinstance(s, tuple) and len(s) == 2:
                r, c = s
                if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                    heatmap_data[r, c] = v[s]
        
        #Calcular límites
        if vmin is None:
            vmin = np.min(heatmap_data)
        if vmax is None:
            vmax = np.max(heatmap_data)
        
        # Crear el mapa de calor con límites específicos
        im = ax.imshow(heatmap_data, cmap='coolwarm', vmin=vmin, vmax=vmax)
        
        # Añadir los valores como texto en cada celda
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                state = (i, j)
                if state in v:
                    # Determinar el color del texto basado en el valor
                    text_color = 'white' if abs(v[state]) > 50 or v[state] < -0.5 else 'black'
                    
                    # Formatear el texto según la magnitud del valor
                    if abs(v[state]) < 10:
                        text = f"{v[state]:.2f}"
                    else:
                        text = f"{v[state]:.1f}"
                    
                    ax.text(j, i, text, 
                        ha="center", va="center", 
                        color=text_color,
                        fontsize=9)
        
        # Añadir una barra de color
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Valor de estado')
        
        # Añadir líneas de cuadrícula
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, grid_shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        # Eliminar los ticks del eje
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Agregar coordenadas a los bordes
        for i in range(grid_shape[0]):
            ax.text(-0.3, i, str(i), ha='right', va='center')
        for j in range(grid_shape[1]):
            ax.text(j, -0.3, str(j), ha='center', va='top')
        
        # Añadir título si se proporciona
        if title:
            ax.set_title(title)
        
        return ax