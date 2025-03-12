"""
Visualizador interactivo de heatmaps para el Ejercicio 3
Muestra directamente en pantalla la evolución de los valores de estado
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

from mdp import get_random_policy
from lake import LakeMDP
from large_lake import large_lake_world
from policy_evaluation._linear import LinearSystemEvaluator
from policy_improvement._standard import StandardPolicyImprover
from policy_iteration._standard import StandardPolicyIteration
from analysis import Analyzer

def create_directional_policy(mdp, direction):
    """
    Crea una política que siempre elige la dirección especificada cuando sea posible.
    """
    def policy(s):
        actions = mdp.get_actions_in_state(s)
        if not actions:  # Si es un estado terminal
            return None
            
        if direction in actions:
            return direction
        return actions[0]  # Si no es posible la dirección preferida
    
    return policy

def run_policy_iterations(mdp, policy_name, init_policy, gamma=0.95, max_iter=10):
    """
    Ejecuta iteración de políticas y devuelve los valores de estado en cada iteración
    """
    print(f"Ejecutando Policy Iteration para política inicial: {policy_name}")
    
    # Lista para almacenar los valores de estado en cada iteración
    state_values_history = []
    
    # Crear evaluador
    evaluator = LinearSystemEvaluator(mdp, gamma)
    evaluator.reset(init_policy)
    
    # Guardar valores iniciales
    state_values_history.append(evaluator.v.copy())
    
    # Crear mejorador e iterador de política
    improver = StandardPolicyImprover()
    policy_iteration = StandardPolicyIteration(init_policy, evaluator, improver)
    
    # Ejecutar iteraciones
    for i in range(max_iter):
        improved = policy_iteration.step()
        
        # Guardar valores de estado después de este paso
        state_values_history.append(evaluator.v.copy())
        
        if not improved:
            print(f"  Política convergió en {i+1} iteraciones")
            break
    
    return state_values_history

def interactive_heatmap_display(mdp, lake_name):
    """
    Muestra una visualización interactiva de los heatmaps para todas las políticas
    direccionales, permitiendo navegar por las iteraciones
    """
    # Ejecutar policy iteration para cada dirección y almacenar resultados
    directions = ['u', 'r', 'd', 'l']
    policy_results = {}
    
    for direction in directions:
        init_policy = create_directional_policy(mdp, direction)
        policy_results[direction] = run_policy_iterations(mdp, direction, init_policy)
    
    # Determinar el número máximo de iteraciones para todas las políticas
    max_iterations = max(len(values) for values in policy_results.values())
    
    # Crear la figura principal
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"{lake_name} Lake - Evolución de los valores de estado", fontsize=16)
    
    # Crear un GridSpec para organizar los heatmaps
    gs = gridspec.GridSpec(2, 2)
    
    # Crear los subplots para cada dirección
    axes = {}
    images = {}
    for i, direction in enumerate(directions):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_title(f"Política inicial: {direction}")
        
        # Crear una matriz para el mapa de calor
        grid_shape = mdp.world.shape
        heatmap_data = np.zeros(grid_shape)
        
        # Inicializar el heatmap con valores vacíos
        img = ax.imshow(heatmap_data, cmap='coolwarm')
        plt.colorbar(img, ax=ax)
        
        # Guardar referencias
        axes[direction] = ax
        images[direction] = img
        
        # Quitar los ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Texto para mostrar la iteración actual
    iteration_text = fig.text(0.5, 0.05, "Iteración: 0", ha='center', fontsize=14)
    
    # Función de actualización para la animación
    def update(iteration):
        iteration_text.set_text(f"Iteración: {iteration}")
        
        for direction in directions:
            # Asegurarse de que tenemos datos para esta iteración
            if iteration < len(policy_results[direction]):
                state_values = policy_results[direction][iteration]
                
                # Actualizar la matriz del heatmap
                grid_shape = mdp.world.shape
                heatmap_data = np.zeros(grid_shape)
                
                # Llenar la matriz con los valores de estado
                for s in state_values:
                    if isinstance(s, tuple) and len(s) == 2:
                        r, c = s
                        if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                            heatmap_data[r, c] = state_values[s]
                
                # Actualizar el heatmap
                images[direction].set_array(heatmap_data)
                images[direction].set_clim(vmin=np.min(heatmap_data), vmax=np.max(heatmap_data))
                
                # Actualizar los textos de valores
                for texts in getattr(axes[direction], '_texts', []):
                    texts.remove()
                axes[direction]._texts = []
                
                for i in range(grid_shape[0]):
                    for j in range(grid_shape[1]):
                        state = (i, j)
                        if state in state_values:
                            text_color = 'white' if abs(state_values[state]) > 50 else 'black'
                            text = axes[direction].text(j, i, f"{state_values[state]:.1f}", 
                                                      ha="center", va="center", 
                                                      color=text_color,
                                                      fontsize=8)
                            if not hasattr(axes[direction], '_texts'):
                                axes[direction]._texts = []
                            axes[direction]._texts.append(text)
        
        return [images[d] for d in directions] + [iteration_text]
    
    # Crear animación
    ani = FuncAnimation(fig, update, frames=range(max_iterations), interval=1000, blit=False)
    
    # Mostrar controles para navegar por las iteraciones
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.show()

def main():
    """Función principal"""
    print("=== Visualizador interactivo de heatmaps para el Ejercicio 3 ===")
    
    # Crear el MDP del lago estándar
    print("\nAnalizando Lake estándar...")
    standard_lake = LakeMDP()
    interactive_heatmap_display(standard_lake, "Standard")
    
    # Crear el MDP del lago grande
    print("\nAnalizando Large Lake...")
    large_lake = LakeMDP(world=large_lake_world)
    interactive_heatmap_display(large_lake, "Large")
    
    print("\nAnalisis completado.")

if __name__ == "__main__":
    main()