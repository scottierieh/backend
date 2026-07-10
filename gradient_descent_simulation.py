
import sys
import json
import numpy as np

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def run_gradient_descent_simulation(learning_rate=0.01, start_x=4.0, start_y=4.0, num_steps=50):
    # 1. Define the function and its gradient
    def f(x, y):
        return x**2 + y**2
    
    def grad_f(x, y):
        return np.array([2*x, 2*y])

    # 2. Initialize
    path = []
    current_pos = np.array([start_x, start_y])
    path.append(current_pos.tolist() + [f(current_pos[0], current_pos[1])])

    # 3. Run Gradient Descent
    for _ in range(num_steps):
        gradient = grad_f(current_pos[0], current_pos[1])
        current_pos = current_pos - learning_rate * gradient
        path.append(current_pos.tolist() + [f(current_pos[0], current_pos[1])])

    # Instead of a VPython visualization, we return the path data
    # that can be used to create a plot on the frontend.
    return {
        "path": path,
        "function_expression": "f(x, y) = x² + y²",
        "parameters": {
            "learning_rate": learning_rate,
            "start_x": start_x,
            "start_y": start_y,
            "num_steps": num_steps
        }
    }


def main():
    try:
        payload = json.load(sys.stdin)
        
        learning_rate = float(payload.get('learning_rate', 0.1))
        start_x = float(payload.get('start_x', 4.0))
        start_y = float(payload.get('start_y', 4.0))
        num_steps = int(payload.get('num_steps', 50))
        
        simulation_data = run_gradient_descent_simulation(learning_rate, start_x, start_y, num_steps)
        
        simulation_data['plot'] = None # No plot generated from backend
        
        print(json.dumps(simulation_data, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
