"""
Ejercicio 2: Policy Iteration
- Implementación de StandardPolicyImprovement y StandardPolicyIteration
- Comparación de políticas óptimas para diferentes valores de gamma
"""

from mdp import get_random_policy
from lake import LakeMDP
from policy_evaluation._linear import LinearSystemEvaluator
from policy_improvement._standard import StandardPolicyImprover
from policy_iteration._standard import StandardPolicyIteration

def debug_mdp(mdp):
    """Imprime información de depuración sobre el MDP"""
    print("\n--- Depuración del MDP ---")
    
    # Verificar estados sin acciones disponibles
    states_without_actions = []
    for s in mdp.states:
        actions = mdp.get_actions_in_state(s)
        if not actions:
            states_without_actions.append(s)
    
    print(f"Estados sin acciones disponibles (probablemente terminales): {states_without_actions}")
    
    # Verificar algunos estados específicos
    sample_states = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for s in sample_states:
        print(f"Estado {s}:")
        print(f"  Recompensa: {mdp.get_reward(s)}")
        print(f"  ¿Es terminal? {mdp.is_terminal_state(s)}")
        print(f"  Acciones disponibles: {mdp.get_actions_in_state(s)}")
    
    print("------------------------\n")

def run_policy_iteration(gamma):
    """Ejecuta la iteración de política con un valor específico de gamma"""
    print(f"\n=== Ejecutando Policy Iteration con gamma = {gamma} ===")
    
    # Crear el entorno del lago
    lake = LakeMDP()
    
    # Depurar el MDP para entender su estructura
    debug_mdp(lake)
    
    # Crear un evaluador de política lineal
    evaluator = LinearSystemEvaluator(lake, gamma)
    
    # Crear un mejorador de política estándar
    improver = StandardPolicyImprover()
    
    # Crear una política inicial aleatoria
    init_policy = get_random_policy(lake, seed=42)
    
    # Crear el algoritmo de iteración de política
    policy_iteration = StandardPolicyIteration(init_policy, evaluator, improver)
    
    # Ejecutar el algoritmo
    final_policy = policy_iteration.run(max_iter=100)  # Limitar a 100 iteraciones por seguridad
    
    # Imprimir la política obtenida
    print(f"\nPolítica óptima para gamma = {gamma}:")
    lake.print_policy(final_policy)
    
    return final_policy

def analyze_policies(policy_095, policy_1):
    """Analiza las diferencias entre las políticas obtenidas con diferentes valores de gamma"""
    print("\n=== Análisis de políticas ===")
    print("Comparación de políticas óptimas con diferentes valores de gamma:")
    
    # Aquí deberías completar con un análisis más detallado basado en los resultados obtenidos
    print("Con gamma = 0.95:")
    print("- El agente valora menos las recompensas futuras")
    print("- Tiende a tomar rutas más directas aunque impliquen más riesgo")
    
    print("\nCon gamma ≈ 1:")
    print("- El agente valora casi igual las recompensas futuras y presentes")
    print("- Tiende a tomar rutas más seguras aunque sean más largas")
    print("- Maximiza la recompensa total acumulada")
    
    print("\nLas diferencias específicas observadas en las políticas son:")
    print("(Completa esto después de ver los resultados)")

def main():
    """Función principal para ejecutar el ejercicio 2"""
    print("=== Ejercicio 2: Policy Iteration ===")
    
    # Valores de gamma a comparar
    gamma_values = [0.95, 1-1e-10]
    
    # Ejecutar con gamma = 0.95
    policy_095 = run_policy_iteration(gamma_values[0])
    
    # Ejecutar con gamma ≈ 1
    policy_1 = run_policy_iteration(gamma_values[1])
    
    # Analizar las diferencias entre las políticas
    analyze_policies(policy_095, policy_1)

if __name__ == "__main__":
    main()