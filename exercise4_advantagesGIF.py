"""
Ejercicio 4: Advantage Function
Visualización de la función de ventaja para el problema del lago
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image

from mdp import get_random_policy
from lake import LakeMDP
from large_lake import large_lake_world
from policy_evaluation._linear import LinearSystemEvaluator
from policy_improvement._standard import StandardPolicyImprover
from policy_iteration._standard import StandardPolicyIteration

def calculate_advantage_function(mdp, v, q):
    """
    Calcula la función de ventaja A(s,a) = Q(s,a) - V(s) para todos los estados y acciones
    """
    advantage = {}
    for s in v:
        if mdp.is_terminal_state(s):
            continue
        advantage[s] = {}
        for a in mdp.get_actions_in_state(s):
            if s in q and a in q[s]:
                advantage[s][a] = q[s][a] - v[s]
    return advantage

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

def create_advantage_grid_visualization(mdp, advantage, iteration, max_adv=10):
    """
    Crea una visualización en forma de grid para la función de ventaja
    siguiendo exactamente el formato requerido por el ejercicio 4
    """
    rows, cols = mdp.world.shape
    fig = plt.figure(figsize=(16, 16))
    
    # Título general
    fig.suptitle(f"Advantages after {iteration} iterations.", fontsize=20, y=0.98)
    
    # Crear un grid para organizar los subplots
    outer_grid = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.4)
    
    # Definir un colormap personalizado: rojo para ventajas negativas, azul para positivas
    colors = [(0.7, 0, 0), (1, 1, 1), (0, 0, 0.7)]  # Rojo, Blanco, Azul
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
    
    # Iterar sobre cada estado (celda de la cuadrícula)
    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            
            # Saltarse estados terminales o que no están en advantage
            if mdp.is_terminal_state(state) or state not in advantage:
                continue
            
            # Crear una subgráfica para este estado
            ax = fig.add_subplot(outer_grid[r, c])
            
            # Crear una matriz 3x3 para las ventajas
            advantage_matrix = np.zeros((3, 3))
            advantage_matrix.fill(np.nan)  # NaN se mostrará como blanco
            
            # Llenar la matriz con los valores de ventaja
            for a, adv in advantage[state].items():
                if a == "u":
                    advantage_matrix[0, 1] = adv
                elif a == "r":
                    advantage_matrix[1, 2] = adv
                elif a == "d":
                    advantage_matrix[2, 1] = adv
                elif a == "l":
                    advantage_matrix[1, 0] = adv
            
            # Visualizar la matriz como un mapa de calor
            im = ax.imshow(advantage_matrix, cmap=cmap, vmin=-max_adv, vmax=max_adv)
            
            # Configurar aspecto
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
    
    # Añadir barras de escala en la derecha
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Advantage')
    # Añadir ticks
    cbar.set_ticks([-10, 0, 10])
    cbar.set_ticklabels(['-10', '0', '10'])
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    return fig

def run_policy_iteration_with_advantage_visualization(mdp, lake_name="standard", output_dir="advantage_images", gamma=0.95, max_iter=6):
    """
    Ejecuta iteración de políticas y genera visualizaciones de la función de ventaja
    para cada iteración, guardando las imágenes para crear un GIF.
    """
    print(f"\n=== Ejecutando Policy Iteration para {lake_name} Lake y visualizando función de ventaja ===")
    
    # Crear directorio para guardar las imágenes si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar con una política aleatoria
    init_policy = get_random_policy(mdp)
    
    # Crear evaluador
    evaluator = LinearSystemEvaluator(mdp, gamma)
    evaluator.reset(init_policy)
    
    # Crear mejorador e iterador de política
    improver = StandardPolicyImprover()
    policy_iteration = StandardPolicyIteration(init_policy, evaluator, improver)
    
    # Lista para almacenar los frames
    frames = []
    
    # Generar visualización para cada iteración
    for i in range(max_iter + 1):  # +1 para incluir la iteración 0
        print(f"  Procesando iteración {i}...")
        
        # Calcular función de ventaja
        v = evaluator.v
        q = evaluator.q
        advantage = calculate_advantage_function(mdp, v, q)
        
        # Crear figura para esta iteración
        fig = create_advantage_grid_visualization(mdp, advantage, i)
        
        # Guardar la figura como imagen
        output_path = f"{output_dir}/{lake_name}_advantage_iter{i}.png"
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"    Imagen guardada: {output_path}")
        
        # Convertir figura a imagen para el GIF
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        
        plt.close(fig)
        
        # Ejecutar un paso de iteración de política si no es la última iteración
        if i < max_iter:
            improved = policy_iteration.step()
            if not improved:
                print(f"    La política convergió en {i+1} iteraciones")
                break
    
    # Guardar como GIF
    gif_path = f"{lake_name}_advantage_evolution.gif"
    print(f"Guardando GIF en {gif_path}...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=1000,  # 1 segundo por frame
        loop=0  # 0 = loop infinito
    )
    
    print(f"GIF generado correctamente: {gif_path}")
    return gif_path, frames

def main():
    """Función principal"""
    print("=== Ejercicio 4: Advantage Function ===")
    
    # Configuración
    gamma = 0.95
    max_iter = 6
    output_dir = "advantage_images"
    
    # Crear el MDP del lago estándar
    standard_lake = LakeMDP()
    standard_gif_path, _ = run_policy_iteration_with_advantage_visualization(
        standard_lake, 
        lake_name="standard", 
        output_dir=output_dir, 
        gamma=gamma, 
        max_iter=max_iter
    )
    
    # Crear el MDP del lago grande
    large_lake = LakeMDP(world=large_lake_world)
    large_gif_path, _ = run_policy_iteration_with_advantage_visualization(
        large_lake, 
        lake_name="large", 
        output_dir=output_dir, 
        gamma=gamma, 
        max_iter=max_iter
    )
    
    print("\n=== Análisis del progreso de optimización ===")
    print("Podemos observar el progreso de optimización en las visualizaciones de la función de ventaja:")
    print("1. En las primeras iteraciones, vemos ventajas positivas (azul) para acciones que llevan hacia el objetivo.")
    print("2. A medida que avanza la iteración de políticas, las ventajas se vuelven más pronunciadas.")
    print("3. Las acciones óptimas muestran ventajas cercanas a cero (blanco) cuando la política converge.")
    print("4. Las acciones sub-óptimas muestran ventajas negativas (rojo) cada vez más fuertes.")
    print("5. Observamos cómo la política se refina hacia el camino óptimo a través de los cambios en las ventajas.")
    print("\nEsto demuestra el principio fundamental de la iteración de políticas: mejorar gradualmente")
    print("hasta encontrar una política que maximice la recompensa esperada.")
    
    print(f"\nProceso completado. GIFs guardados en: {standard_gif_path} y {large_gif_path}")

if __name__ == "__main__":
    main()