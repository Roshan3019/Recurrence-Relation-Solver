from flask import Flask, request, jsonify
from flask_cors import CORS
import sympy as sp
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Helper function to parse initial conditions
def parse_initial_conditions(conditions_str):
    conditions = {}
    pattern = r'a\((\d+)\)\s*=\s*([-]?\d*\.?\d*)'
    matches = re.findall(pattern, conditions_str)
    for index, value in matches:
        conditions[int(index)] = float(value)
    return conditions

# Helper function to generate chart data
def generate_chart_data(solution, initial_conditions, max_n=10):
    n = sp.Symbol('n')
    chart_data = []
    for i in range(1, max_n + 1):
        try:
            value = solution.subs(n, i).evalf()
            if sp.im(value) == 0:  # Ensure real numbers
                chart_data.append(float(value))
            else:
                chart_data.append(0)  # Handle complex results
        except:
            chart_data.append(0)  # Handle undefined cases
    return chart_data

# Simplify LaTeX to readable string 
# def simplify_latex_to_string(latex_str):
#     latex_str = latex_str.replace(r'\cdot', '*').replace(r'\{', '').replace(r'\}', '')
#     return latex_str.replace(r'^{', '^').replace(r'}', '')
# Simplify LaTeX to readable string
def simplify_latex_to_string(latex_str):
    # Remove LaTeX-specific symbols
    latex_str = latex_str.replace(r'\cdot', '*').replace(r'\{', '').replace(r'\}', '')
    latex_str = latex_str.replace(r'^{', '^').replace(r'}', '')
    # Handle fractions (e.g., \frac{n \left(n + 1\right)}{2} -> n * (n + 1) / 2)
    latex_str = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'(\1) / \2', latex_str)
    # Remove unnecessary spaces and LaTeX artifacts
    latex_str = latex_str.replace(r'\left', '').replace(r'\right', '').replace(r'\,', '')
    latex_str = re.sub(r'\s+', ' ', latex_str).strip()
    return latex_str

# Main endpoint to solve recurrence relations
@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    relation_type = data.get('relationType')
    equation = data.get('equation')
    initial_conditions_str = data.get('initialConditions')

    if not all([relation_type, equation, initial_conditions_str]):
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        initial_conditions = parse_initial_conditions(initial_conditions_str)
        if not initial_conditions:
            return jsonify({'error': 'Invalid initial conditions'}), 400

        n = sp.Symbol('n')
        a = sp.Function('a')
        steps = []

        if relation_type == 'linear' or relation_type == 'homogeneous':
            eq_str = equation.replace('a(n)', 'a(n)').replace('=', '-')
            eq = sp.parse_expr(eq_str, local_dict={'a': a, 'n': n})
            steps.append(f"Parsed equation: {eq} = 0")
            
            sol = sp.rsolve(eq, a(n), initial_conditions)
            steps.append(f"Characteristic equation solved: {sol}")
            
            for idx, val in initial_conditions.items():
                steps.append(f"Verification: a({idx}) = {val}, computed: {sol.subs(n, idx).evalf()}")
            
            final_eq = simplify_latex_to_string(sp.latex(sol))
            chart_data = generate_chart_data(sol, initial_conditions)

        elif relation_type == 'non-homogeneous':
            eq_str = equation.replace('a(n)', 'a(n)').replace('=', '-')
            eq = sp.parse_expr(eq_str, local_dict={'a': a, 'n': n})
            steps.append(f"Parsed equation: {eq} = 0")
            
            homo_eq = eq.subs({a(n-1): a(n-1), a(n-2): a(n-2) if 'a(n-2)' in str(eq) else 0})
            homo_sol = sp.rsolve(homo_eq, a(n))
            steps.append(f"Homogeneous solution: a_h(n) = {homo_sol}")
            
            particular_sol = sp.rsolve(eq, a(n), initial_conditions)
            steps.append(f"Particular solution: a_p(n) = {particular_sol - homo_sol}")
            
            full_sol = particular_sol
            steps.append(f"Full solution: a(n) = {full_sol}")
            
            final_eq = simplify_latex_to_string(sp.latex(full_sol))
            chart_data = generate_chart_data(full_sol, initial_conditions)

        elif relation_type == 'divide':
            steps.append("Applying Master Theorem for a(n) = a*T(n/b) + f(n)")
            a = float(re.search(r'(\d+)\*a', equation).group(1))
            b = 2  # Assuming n/2
            f_n = equation.split('+')[1].strip()
            steps.append(f"a = {a}, b = {b}, f(n) = {f_n}")
            
            log_b_a = np.log2(a)
            if 'n' in f_n:
                f_n_degree = 1  # Assuming linear f(n) = n
                steps.append(f"log_b(a) = {log_b_a}, degree of f(n) = {f_n_degree}")
                
                if f_n_degree < log_b_a:
                    sol = f"O(n^{log_b_a})"
                elif f_n_degree == log_b_a:
                    sol = f"O(n^{log_b_a} * log(n))"
                else:
                    sol = f"O(n^{f_n_degree})"
                steps.append(f"Solution via Master Theorem: {sol}")
            else:
                sol = f"O(n^{log_b_a})"
                steps.append(f"Solution: {sol}")
            
            final_eq = sol
            chart_data = [i * np.log2(i) for i in range(1, 11)]  # Approximate for visualization

        elif relation_type == 'substitution':
            steps.append("Using substitution method")
            eq_str = equation.replace('a(n)', 'a(n)').replace('=', '-')
            eq = sp.parse_expr(eq_str, local_dict={'a': a, 'n': n})
            initial_dict = {a(i): val for i, val in initial_conditions.items()}
            sol = sp.rsolve(eq, a(n), initial_dict)
            steps.append(f"Substituted and solved: a(n) = {sol}")
            
            final_eq = simplify_latex_to_string(sp.latex(sol))
            chart_data = generate_chart_data(sol, initial_conditions)

        else:
            return jsonify({'error': 'Invalid relation type'}), 400

        solution_html = "<ol>" + "".join(f"<li>{step}</li>" for step in steps) + "</ol>"

        return jsonify({
            'solution': solution_html,
            'final_equation': final_eq,
            'chart_data': chart_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for examples
@app.route('/examples', methods=['GET'])
@app.route('/examples', methods=['GET'])
def get_examples():
    examples = [
        {
            'type': 'linear',
            'equation': 'a(n) = 2*a(n-1) + 3',
            'initial_conditions': 'a(0)=1',
            'solution': 'a(n) = 2^n + 3'
        },
        {
            'type': 'divide',
            'equation': 'a(n) = 2*a(n/2) + n',
            'initial_conditions': 'a(1)=1',
            'solution': 'a(n) = n * log(n) + n'
        },
        {
            'type': 'substitution',
            'equation': 'a(n) = a(n-1) + n',
            'initial_conditions': 'a(0)=0',
            'solution': 'a(n) = (n * (n + 1)) / 2'
        },
        {
            'type': 'homogeneous',
            'equation': 'a(n) = 2*a(n-1)',
            'initial_conditions': 'a(0)=1',
            'solution': 'a(n) = 2^n'
        },
        {
            'type': 'non-homogeneous',
            'equation': 'a(n) = 2*a(n-1) + n',
            'initial_conditions': 'a(0)=1',
            'solution': 'a(n) = 2^n + n * n'
        }
    ]
    return jsonify(examples)

if __name__ == '__main__':
    app.run(port=8000, debug=True)