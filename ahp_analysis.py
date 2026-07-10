import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class AHPAnalysis:
    """
    Analytic Hierarchy Process (AHP) Analysis Class
    Supports both: 
    1. Criteria-only AHP (no alternatives)
    2. Full AHP with alternatives
    """
    
    def __init__(self, criteria_names, alternative_names=None):
        """
        Initialize AHP analysis
        
        Parameters:
        - criteria_names: List of criteria names
        - alternative_names: List of alternative names (optional)
        """
        self.criteria_names = criteria_names
        self.alternative_names = alternative_names
        self.n_criteria = len(criteria_names)
        self.has_alternatives = alternative_names is not None
        
        if self.has_alternatives:
            self.n_alternatives = len(alternative_names)
            
        # Initialize matrices
        self.criteria_matrix = np.ones((self.n_criteria, self.n_criteria))
        
        if self.has_alternatives:
            # Alternative comparison matrices for each criterion
            self.alternative_matrices = {}
            for criterion in criteria_names:
                self.alternative_matrices[criterion] = np.ones((self.n_alternatives, self.n_alternatives))
    
    def set_criteria_matrix(self, matrix):
        """Set pairwise comparison matrix for criteria"""
        self.criteria_matrix = np.array(matrix, dtype=np.float64)
        print(f"DEBUG: Criteria matrix set:", file=sys.stderr)
        print(self.criteria_matrix, file=sys.stderr)
    
    def set_alternative_matrix(self, criterion_name, matrix):
        """Set pairwise comparison matrix for alternatives under a criterion"""
        if not self.has_alternatives:
            raise ValueError("No alternatives defined")
        self.alternative_matrices[criterion_name] = np.array(matrix, dtype=np.float64)

    def calculate_weights(self, matrix):
        """Calculate priority weights using eigenvalue method"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_eigenvalue_index = np.argmax(eigenvalues.real)
        max_eigenvalue = eigenvalues[max_eigenvalue_index].real
        priority_vector = eigenvectors[:, max_eigenvalue_index].real
        weights = priority_vector / np.sum(priority_vector)
        
        print(f"DEBUG: Eigenvalues: {eigenvalues}", file=sys.stderr)
        print(f"DEBUG: Max eigenvalue: {max_eigenvalue}", file=sys.stderr)
        print(f"DEBUG: Weights: {weights}", file=sys.stderr)
        
        return weights, max_eigenvalue
    
    def calculate_consistency_ratio(self, matrix, max_eigenvalue):
        """Calculate Consistency Ratio (CR)"""
        RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
              6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        
        n = len(matrix)
        CI = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
        CR = CI / RI[n] if n in RI and RI[n] != 0 else 0
        
        print(f"DEBUG CR Calculation:", file=sys.stderr)
        print(f"  Matrix size n = {n}", file=sys.stderr)
        print(f"  Max eigenvalue = {max_eigenvalue}", file=sys.stderr)
        print(f"  CI = ({max_eigenvalue} - {n}) / ({n} - 1) = {CI}", file=sys.stderr)
        print(f"  RI[{n}] = {RI.get(n, 'N/A')}", file=sys.stderr)
        print(f"  CR = {CI} / {RI.get(n, 0)} = {CR}", file=sys.stderr)
        
        return CR, CI
    
    def analyze(self):
        """Perform complete AHP analysis"""
        print("\n" + "="*60, file=sys.stderr)
        print("ANALYZING CRITERIA", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        # Calculate criteria weights
        self.criteria_weights, self.criteria_lambda_max = self.calculate_weights(self.criteria_matrix)
        self.criteria_CR, self.criteria_CI = self.calculate_consistency_ratio(
            self.criteria_matrix, self.criteria_lambda_max)
        
        if self.has_alternatives:
            print("\n" + "="*60, file=sys.stderr)
            print("ANALYZING ALTERNATIVES", file=sys.stderr)
            print("="*60, file=sys.stderr)
            
            # Calculate alternative weights for each criterion
            self.alternative_weights = {}
            self.alternative_CR = {}
            self.alternative_CI = {}
            self.alternative_lambda_max = {}
            
            for criterion in self.criteria_names:
                print(f"\n--- Criterion: {criterion} ---", file=sys.stderr)
                weights, lambda_max = self.calculate_weights(
                    self.alternative_matrices[criterion])
                CR, CI = self.calculate_consistency_ratio(
                    self.alternative_matrices[criterion], lambda_max)
                
                self.alternative_weights[criterion] = weights
                self.alternative_lambda_max[criterion] = lambda_max
                self.alternative_CR[criterion] = CR
                self.alternative_CI[criterion] = CI
            
            self.calculate_final_scores()
    
    def calculate_final_scores(self):
        """Calculate final scores by combining criteria and alternative weights"""
        if not self.has_alternatives:
            return
        
        self.final_scores = np.zeros(self.n_alternatives)
        
        for i, criterion in enumerate(self.criteria_names):
            criterion_weight = self.criteria_weights[i]
            alternative_weight = self.alternative_weights[criterion]
            self.final_scores += criterion_weight * alternative_weight
        
        self.ranking = np.argsort(self.final_scores)[::-1]
        
        print(f"\nDEBUG Final Scores: {self.final_scores}", file=sys.stderr)
        print(f"DEBUG Ranking: {[self.alternative_names[i] for i in self.ranking]}", file=sys.stderr)

    def get_results(self):
        """Return analysis results as a dictionary with JSON-safe types"""
        results = {
            'criteria_analysis': {
                'weights': {name: float(weight) for name, weight in zip(self.criteria_names, self.criteria_weights)},
                'consistency': {
                    'lambda_max': float(self.criteria_lambda_max),
                    'CI': float(self.criteria_CI),
                    'CR': float(self.criteria_CR),
                    'is_consistent': bool(self.criteria_CR < 0.1)
                }
            }
        }
        
        if self.has_alternatives:
            results['alternative_weights_by_criterion'] = {}
            for criterion in self.criteria_names:
                results['alternative_weights_by_criterion'][criterion] = {
                    'weights': {name: float(weight) for name, weight in zip(self.alternative_names, self.alternative_weights[criterion])},
                    'consistency': {
                        'lambda_max': float(self.alternative_lambda_max[criterion]),
                        'CI': float(self.alternative_CI[criterion]),
                        'CR': float(self.alternative_CR[criterion]),
                        'is_consistent': bool(self.alternative_CR[criterion] < 0.1)
                    }
                }
            
            # Final scores as sorted list
            results['final_scores'] = [
                {'name': self.alternative_names[i], 'score': float(self.final_scores[i])}
                for i in self.ranking
            ]
            results['ranking'] = [self.alternative_names[i] for i in self.ranking]

        return results

def geometric_mean_of_matrices(matrices):
    """Calculates the element-wise geometric mean of a list of matrices."""
    if not matrices:
        return None
    
    print(f"DEBUG: Calculating geometric mean of {len(matrices)} matrices", file=sys.stderr)
    
    matrices_array = np.array(matrices, dtype=np.float64)
    
    # Handle any invalid values
    matrices_array = np.where(matrices_array <= 0, 1, matrices_array)
    
    log_matrices = np.log(matrices_array)
    mean_log_matrix = np.mean(log_matrices, axis=0)
    geo_mean_matrix = np.exp(mean_log_matrix)
    
    # Ensure symmetry
    for i in range(geo_mean_matrix.shape[0]):
        for j in range(i, geo_mean_matrix.shape[1]):
            if i == j:
                geo_mean_matrix[i, j] = 1.0
            else:
                geo_mean_matrix[j, i] = 1.0 / geo_mean_matrix[i, j]
    
    print(f"DEBUG: Result matrix:", file=sys.stderr)
    print(geo_mean_matrix, file=sys.stderr)
                
    return geo_mean_matrix

def main():
    try:
        input_data = sys.stdin.read()
        
        if not input_data.strip():
            raise ValueError("Empty input received")
        
        payload = json.loads(input_data)
        
        hierarchy = payload.get('hierarchy')
        matrices_by_respondent = payload.get('matrices')
        alternatives = payload.get('alternatives')
        goal = payload.get('goal', 'Goal')

        print("="*60, file=sys.stderr)
        print("AHP ANALYSIS STARTED", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"Number of matrix sets: {len(matrices_by_respondent)}", file=sys.stderr)
        print(f"Matrix keys: {list(matrices_by_respondent.keys())}", file=sys.stderr)

        if not hierarchy or not isinstance(hierarchy, list) or len(hierarchy) == 0:
            raise ValueError(f"Invalid or missing hierarchy data. Received: {hierarchy}")
        
        if not matrices_by_respondent or not isinstance(matrices_by_respondent, dict):
            raise ValueError("Invalid or missing matrices data")

        criteria_nodes = []
        if 'nodes' in hierarchy[0]:
            for node in hierarchy[0]['nodes']:
                criteria_nodes.append(node['name'])

        if not criteria_nodes:
            raise ValueError("No criteria found in hierarchy structure.")

        print(f"Criteria found: {criteria_nodes}", file=sys.stderr)
        print(f"Alternatives: {alternatives}", file=sys.stderr)

        has_alternatives = bool(alternatives and len(alternatives) > 0)
        
        ahp = AHPAnalysis(criteria_nodes, alternatives if has_alternatives else None)
        
        agg_matrices = {}
        for key, matrix_list in matrices_by_respondent.items():
            if matrix_list and len(matrix_list) > 0:
                print(f"\nAggregating matrices for key: {key}", file=sys.stderr)
                agg_matrices[key] = geometric_mean_of_matrices(matrix_list)

        if 'criteria' in agg_matrices:
             ahp.set_criteria_matrix(agg_matrices['criteria'])
        elif 'goal' in agg_matrices:
            ahp.set_criteria_matrix(agg_matrices['goal'])
        else:
            raise ValueError("Criteria comparison matrix for 'goal' or 'criteria' not found in matrices")

        if has_alternatives:
            for criterion in criteria_nodes:
                matrix_key = f"alt_{criterion}"
                if matrix_key in agg_matrices:
                    ahp.set_alternative_matrix(criterion, agg_matrices[matrix_key])

        ahp.analyze()
        results_data = ahp.get_results()

        response = {"results": results_data}
        
        print("\n" + "="*60, file=sys.stderr)
        print("ANALYSIS COMPLETE", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        
        print(json.dumps(response, cls=NumpyEncoder, ensure_ascii=False, indent=2))

    except Exception as e:
        error_response = {
            "error": "Analysis failed",
            "details": str(e),
            "type": type(e).__name__
        }
        print(json.dumps(error_response, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()