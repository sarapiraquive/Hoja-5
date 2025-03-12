"""
Ejercicio 3: Plot Development of State Values
- Implementación de métodos para visualizar la evolución de valores de estado
- Ejecución de Policy Iteration con diferentes políticas iniciales direccionales
"""

import matplotlib.pyplot as plt
import numpy as np
import os

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
    
    :param mdp: El MDP del problema
    :param direction: La dirección preferida ('u', 'r', 'd', 'l')
    :return: Una función de política que mapea estados a acciones
    """
    def policy(s):
        actions = mdp.get_actions_in_state(s)
        if not actions:  # Si es un estado terminal
            return None
            
        # Priorizar la dirección especificada
        if direction in actions:
            return direction
        # Si no es posible la dirección preferida, elegir cualquier acción disponible
        return actions[0]
    
    return policy

def run_policy_iteration_analysis(mdp, analyzer, direction, gamma=0.95, max_iter=10):
    """
    Ejecuta iteración de políticas con una política inicial direccional y registra
    la evolución de los valores de estado.
    
    :param mdp: El MDP del problema
    :param analyzer: Objeto Analyzer para registrar los valores de estado
    :param direction: Dirección inicial ('u', 'r', 'd', 'l')
    :param gamma: Factor de descuento
    :param max_iter: Número máximo de iteraciones
    :return: La política final
    """
    print(f"\nEjecutando Policy Iteration con política inicial direccional: {direction}")
    
    # Crear política inicial direccional
    init_policy = create_directional_policy(mdp, direction)
    
    # Iniciar un nuevo run en el analizador
    analyzer.new_run(direction)
    
    # Crear evaluador de política
    evaluator = LinearSystemEvaluator(mdp, gamma)
    evaluator.reset(init_policy)
    
    # Registrar valores de estado iniciales
    analyzer.add_state_value_estimates(evaluator.v)
    
    # Crear mejorador de política
    improver = StandardPolicyImprover()
    
    # Crear algoritmo de iteración de políticas
    policy_iteration = StandardPolicyIteration(init_policy, evaluator, improver)
    
    # Ejecutar iteraciones y registrar valores de estado
    for i in range(max_iter):
        improved = policy_iteration.step()
        analyzer.add_state_value_estimates(evaluator.v)
        
        if not improved:
            print(f"  Política convergió en {i+1} iteraciones")
            break
    
    return policy_iteration.policy_improver.policy

def analyze_lake(lake, lake_name, gamma=0.95, max_iter=10):
    """
    Analiza un MDP de lago ejecutando iteración de políticas con diferentes
    políticas iniciales direccionales.
    
    :param lake: El MDP del lago a analizar
    :param lake_name: Nombre del lago para identificación
    :param gamma: Factor de descuento
    :param max_iter: Número máximo de iteraciones
    """
    print(f"\n=== Analizando {lake_name} Lake ===")
    
    # Crear analizador
    analyzer = Analyzer(lake)
    
    # Ejecutar iteración de políticas para cada dirección
    for direction in ['u', 'r', 'd', 'l']:
        run_policy_iteration_analysis(lake, analyzer, direction, gamma, max_iter)
    
    # Crear figura para las gráficas
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Graficar evolución del valor del estado inicial
    analyzer.plot_state_value_estimates_of_init_state_over_time(ax=axes[0])
    axes[0].set_title(f'Evolución del valor del estado inicial - {lake_name} Lake')
    
    # Graficar evolución del valor promedio de estados
    analyzer.plot_avg_state_value_estimates_over_time(ax=axes[1])
    axes[1].set_title(f'Evolución del valor promedio de estados - {lake_name} Lake')
    
    # Ajustar y guardar la figura
    plt.tight_layout()
    plt.savefig(f'{lake_name}_lake_state_values_evolution.png')
    plt.show()
    
    return analyzer

def main():
    """Función principal para ejecutar el ejercicio 3"""
    print("=== Ejercicio 3: Plot Development of State Values ===")
    
    # Configuración general
    gamma = 0.95
    max_iter = 10
    
    # Analizar lago estándar (4x4)
    standard_lake = LakeMDP()
    standard_analyzer = analyze_lake(standard_lake, "Standard", gamma, max_iter)
    
    # Analizar lago grande (10x10)
    large_lake = LakeMDP(world=large_lake_world)
    large_analyzer = analyze_lake(large_lake, "Large", gamma, max_iter)
    
    print("\nAnálisis completado. Las gráficas han sido guardadas.")
    
if __name__ == "__main__":
    main()