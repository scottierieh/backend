import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.contingency_tables import mcnemar
import warnings
from typing import Dict, List, Tuple, Optional, Union
import io
import base64
import math

warnings.filterwarnings('ignore')

# Set seaborn style globally
sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

class NonParametricTests:
    def __init__(self, data: pd.DataFrame):
        self.data = pd.DataFrame(data)
        self.results = {}

    def mann_whitney_u_test(self, group_col: str, value_col: str, groups: List[str] = None, alternative: str = 'two-sided', alpha: float = 0.05) -> Dict:
        if groups is None:
            unique_groups = self.data[group_col].unique()
            if len(unique_groups) != 2:
                raise ValueError(f"Found {len(unique_groups)} groups, need exactly 2 for Mann-Whitney U test")
            groups = unique_groups.tolist()
        
        group1_data = self.data[self.data[group_col] == groups[0]][value_col].dropna()
        group2_data = self.data[self.data[group_col] == groups[1]][value_col].dropna()
        
        if len(group1_data) == 0 or len(group2_data) == 0:
            raise ValueError("One or both groups have no data.")

        statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative=alternative)
        
        n1, n2 = len(group1_data), len(group2_data)
        
        combined_data = np.concatenate([group1_data, group2_data])
        ranks = rankdata(combined_data)
        R1 = np.sum(ranks[:n1])
        R2 = np.sum(ranks[n1:])
        
        U1 = R1 - n1 * (n1 + 1) / 2
        U2 = R2 - n2 * (n2 + 1) / 2
        U_stat = min(U1, U2)
        
        # Calculate effect size r
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) if (n1+n2) > 0 else 0
        z_score = (U_stat - mean_u) / std_u if std_u > 0 else 0
        r = abs(z_score) / np.sqrt(n1 + n2) if (n1 + n2) > 0 else 0
        
        desc_stats = {
            groups[0]: {
                'n': n1, 
                'median': float(np.median(group1_data)), 
                'mean': float(np.mean(group1_data)),
                'std': float(np.std(group1_data, ddof=1)) if n1 > 1 else 0,
                'mean_rank': R1 / n1 if n1 > 0 else 0
            },
            groups[1]: {
                'n': n2, 
                'median': float(np.median(group2_data)), 
                'mean': float(np.mean(group2_data)),
                'std': float(np.std(group2_data, ddof=1)) if n2 > 1 else 0,
                'mean_rank': R2 / n2 if n2 > 0 else 0
            }
        }
        
        is_significant = p_value < alpha
        effect_size_interp = self._interpret_effect_size(r, 'correlation')
        
        result = {
            'test_type': 'Mann-Whitney U Test', 
            'statistic': statistic, 
            'p_value': p_value,
            'U': U_stat, 
            'R1': R1, 
            'R2': R2, 
            'n1': n1, 
            'n2': n2,
            'z_score': z_score,
            'effect_size': r, 
            'effect_size_interpretation': effect_size_interp, 
            'alpha': alpha,
            'is_significant': is_significant, 
            'alternative': alternative, 
            'groups': groups,
            'descriptive_stats': desc_stats,
            'interpretation': self._interpret_mann_whitney(is_significant, groups, p_value, group_col, value_col, desc_stats, z_score, alpha),
            'group_col': group_col, 
            'value_col': value_col
        }
        return result

    def wilcoxon_signed_rank_test(self, var1: str, var2: str, alternative: str = 'two-sided', alpha: float = 0.05) -> Dict:
        clean_data = self.data[[var1, var2]].dropna()
        if len(clean_data) < 10:
             raise ValueError("Wilcoxon test requires at least 10 complete pairs.")
        
        data1 = clean_data[var1]
        data2 = clean_data[var2]

        if len(data1) == 0: 
            raise ValueError("No valid paired observations found")
        
        differences = data1 - data2
        non_zero_diffs = differences[differences != 0]
        
        statistic, p_value = stats.wilcoxon(data1, data2, alternative=alternative, zero_method='pratt')
        
        abs_diffs = np.abs(non_zero_diffs)
        ranks = rankdata(abs_diffs)
        W_plus = np.sum(ranks[non_zero_diffs > 0])
        W_minus = np.sum(ranks[non_zero_diffs < 0])
        
        n_diff = len(non_zero_diffs)
        mean_W = n_diff * (n_diff + 1) / 4
        std_W = np.sqrt(n_diff * (n_diff + 1) * (2 * n_diff + 1) / 24)
        z_score = (statistic - mean_W) / std_W if std_W > 0 else 0

        r = z_score / np.sqrt(n_diff) if n_diff > 0 else 0
        
        desc_stats = {
            var1: {
                'n': len(data1), 
                'median': float(np.median(data1)), 
                'mean': float(np.mean(data1)), 
                'std': float(np.std(data1, ddof=1)) if len(data1) > 1 else 0
            },
            var2: {
                'n': len(data2), 
                'median': float(np.median(data2)), 
                'mean': float(np.mean(data2)), 
                'std': float(np.std(data2, ddof=1)) if len(data2) > 1 else 0
            }
        }
        
        is_significant = p_value < alpha
        effect_size_interp = self._interpret_effect_size(r, 'correlation')
        
        result = {
            'test_type': 'Wilcoxon Signed-Rank Test', 
            'statistic': statistic, 
            'p_value': p_value, 
            'z_score': z_score,
            'W_plus': W_plus, 
            'W_minus': W_minus, 
            'n_pairs': len(data1), 
            'effect_size': r,
            'effect_size_interpretation': effect_size_interp, 
            'alpha': alpha, 
            'is_significant': is_significant,
            'alternative': alternative, 
            'variables': [var1, var2], 
            'descriptive_stats': desc_stats,
            'interpretation': self._interpret_wilcoxon(is_significant, var1, var2, p_value, z_score, desc_stats, alpha)
        }
        
        return result

    def kruskal_wallis_test(self, group_col: str, value_col: str, alpha: float = 0.05) -> Dict:
        groups_data = [self.data[self.data[group_col] == name][value_col].dropna() for name in self.data[group_col].unique()]
        if len(groups_data) < 3: 
            raise ValueError("Kruskal-Wallis test requires at least 3 groups")

        statistic, p_value = stats.kruskal(*groups_data)
        
        n_total = sum(len(g) for g in groups_data)
        k = len(groups_data)
        df = k - 1
        
        epsilon_squared = (statistic - k + 1) / (n_total - k) if (n_total - k) > 0 else 0
        
        # Calculate descriptive statistics for each group
        group_names = self.data[group_col].unique()
        desc_stats = {}
        for i, group_name in enumerate(group_names):
            group_data = groups_data[i]
            desc_stats[group_name] = {
                'n': len(group_data),
                'median': float(np.median(group_data)) if len(group_data) > 0 else 0,
                'mean': float(np.mean(group_data)) if len(group_data) > 0 else 0,
                'std': float(np.std(group_data, ddof=1)) if len(group_data) > 1 else 0
            }
        
        is_significant = p_value < alpha
        effect_size_interp = self._interpret_epsilon_squared(epsilon_squared)
        
        result = {
            'test_type': 'Kruskal-Wallis H Test', 
            'statistic': statistic, 
            'p_value': p_value, 
            'df': df,
            'n_groups': k, 
            'n_total': n_total, 
            'effect_size': epsilon_squared,
            'effect_size_interpretation': effect_size_interp, 
            'alpha': alpha, 
            'is_significant': is_significant,
            'descriptive_stats': desc_stats,
            'interpretation': self._interpret_kruskal_wallis(is_significant, group_names, effect_size_interp, epsilon_squared, p_value, alpha, statistic, df, group_col, value_col),
            'group_col': group_col, 
            'value_col': value_col
        }
        return result

    def friedman_test(self, variables: List[str], alpha: float = 0.05) -> Dict:
        if len(variables) < 3: 
            raise ValueError("Friedman test requires at least 3 variables")
        
        data_matrix = self.data[variables].dropna().values
        if data_matrix.shape[0] == 0: 
            raise ValueError("No complete observations found")

        statistic, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(len(variables))])
        
        n_subjects, k_conditions = data_matrix.shape
        df = k_conditions - 1
        W = statistic / (n_subjects * (k_conditions - 1)) if n_subjects > 0 and k_conditions > 1 else 0
        
        # Calculate descriptive statistics for each variable
        desc_stats = {}
        for i, var in enumerate(variables):
            var_data = data_matrix[:, i]
            desc_stats[var] = {
                'n': len(var_data),
                'median': float(np.median(var_data)),
                'mean': float(np.mean(var_data)),
                'std': float(np.std(var_data, ddof=1)) if len(var_data) > 1 else 0
            }
        
        is_significant = p_value < alpha
        effect_size_interp = self._interpret_kendall_w(W)

        result = {
            'test_type': 'Friedman Test', 
            'statistic': statistic, 
            'p_value': p_value, 
            'df': df,
            'n_subjects': n_subjects, 
            'k_conditions': k_conditions, 
            'effect_size': W,
            'effect_size_interpretation': effect_size_interp, 
            'alpha': alpha, 
            'is_significant': is_significant,
            'descriptive_stats': desc_stats,
            'interpretation': self._interpret_friedman(is_significant, variables, effect_size_interp, W, p_value, alpha, statistic, df),
            'variables': variables
        }
        return result
        
    def mcnemar_test(self, var1: str, var2: str, alpha: float = 0.05) -> Dict:
        clean_data = self.data[[var1, var2]].dropna()
        if len(clean_data) < 10:
            raise ValueError("Not enough paired data for McNemar's test (minimum 10 pairs required).")
            
        contingency_table = pd.crosstab(clean_data[var1], clean_data[var2])
        
        # Ensure it's a 2x2 table
        if contingency_table.shape != (2, 2):
            raise ValueError("McNemar's test requires a 2x2 contingency table. Ensure both variables are binary.")
            
        result = mcnemar(contingency_table, exact=True)  # Use exact test for small samples
        
        statistic = result.statistic
        p_value = result.pvalue
        
        is_significant = p_value < alpha
        
        # Enhanced interpretation similar to other tests
        p_val_text = f"p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        
        interpretation = (
            f"A McNemar's test was conducted to examine the change in paired proportions between '{var1}' and '{var2}'.\n\n"
            f"The results indicated a {'statistically significant' if is_significant else 'not statistically significant'} "
            f"change in proportions, χ² = {statistic:.2f}, {p_val_text}."
        )
        
        if is_significant:
            interpretation += f"\n\nThis suggests that there is a significant change in the distribution of outcomes between the two related conditions."

        result_dict = {
            'test_type': "McNemar's Test",
            'statistic': statistic,
            'p_value': p_value,
            'contingency_table': contingency_table.to_dict(),
            'alpha': alpha,
            'is_significant': is_significant,
            'interpretation': interpretation,
            'variables': [var1, var2]
        }
        return result_dict

    def _interpret_effect_size(self, r, type): 
        if abs(r) >= 0.5:
            return {'level': 'large', 'text': 'Large effect'}
        elif abs(r) >= 0.3:
            return {'level': 'medium', 'text': 'Medium effect'}
        elif abs(r) >= 0.1:
            return {'level': 'small', 'text': 'Small effect'}
        else:
            return {'level': 'negligible', 'text': 'Negligible effect'}
    
    def _interpret_epsilon_squared(self, es):
        if es >= 0.14:
            return {'level': 'large', 'text': 'Large effect'}
        elif es >= 0.06:
            return {'level': 'medium', 'text': 'Medium effect'}
        elif es >= 0.01:
            return {'level': 'small', 'text': 'Small effect'}
        else:
            return {'level': 'negligible', 'text': 'Negligible effect'}
    
    def _interpret_kendall_w(self, w):
        if w >= 0.3:
            return {'level': 'strong', 'text': 'Strong concordance'}
        elif w >= 0.1:
            return {'level': 'moderate', 'text': 'Moderate concordance'}
        else:
            return {'level': 'weak', 'text': 'Weak concordance'}
    
    def _interpret_mann_whitney(self, is_sig, groups, p, group_col, value_col, desc_stats, z_score, alpha):
        p_val_text = f"p < .001" if p < 0.001 else f"p = {p:.3f}"
        
        g1_name, g2_name = groups[0], groups[1]
        g1_stats = desc_stats[g1_name]
        g2_stats = desc_stats[g2_name]
        
        interpretation = (
            f"A Mann-Whitney U test was conducted to determine if there was a statistically significant difference "
            f"in '{value_col}' between '{g1_name}' and '{g2_name}' groups.\n\n"
            f"The results indicated a {'statistically significant' if is_sig else 'not statistically significant'} "
            f"difference, U = {self.results.get('U', 0):.1f}, z = {z_score:.2f}, {p_val_text}."
        )
        
        if is_sig:
            higher_group = g1_name if g1_stats['mean_rank'] > g2_stats['mean_rank'] else g2_name
            lower_group = g2_name if g1_stats['mean_rank'] > g2_stats['mean_rank'] else g1_name
            higher_stats = g1_stats if g1_stats['mean_rank'] > g2_stats['mean_rank'] else g2_stats
            lower_stats = g2_stats if g1_stats['mean_rank'] > g2_stats['mean_rank'] else g1_stats
            
            interpretation += (
                f"\n\nThe '{higher_group}' group (Mdn = {higher_stats['median']:.2f}, Mean Rank = {higher_stats['mean_rank']:.2f}) "
                f"had significantly higher scores than the '{lower_group}' group "
                f"(Mdn = {lower_stats['median']:.2f}, Mean Rank = {lower_stats['mean_rank']:.2f})."
            )
        
        return interpretation.strip()

    def _interpret_wilcoxon(self, is_sig, v1, v2, p, z_score, desc_stats, alpha):
        p_val_text = f"p < .001" if p < 0.001 else f"p = {p:.3f}"
        
        v1_stats = desc_stats[v1]
        v2_stats = desc_stats[v2]
        
        interpretation = (
            f"A Wilcoxon Signed-Rank test was conducted to determine if there was a statistically significant "
            f"difference between paired observations of '{v1}' and '{v2}'.\n\n"
            f"The results indicated a {'statistically significant' if is_sig else 'not statistically significant'} "
            f"difference, z = {z_score:.2f}, {p_val_text}."
        )
        
        if is_sig:
            higher_var = v1 if v1_stats['median'] > v2_stats['median'] else v2
            lower_var = v2 if v1_stats['median'] > v2_stats['median'] else v1
            higher_stats = v1_stats if v1_stats['median'] > v2_stats['median'] else v2_stats
            lower_stats = v2_stats if v1_stats['median'] > v2_stats['median'] else v1_stats
            
            interpretation += (
                f"\n\n'{higher_var}' (Mdn = {higher_stats['median']:.2f}, M = {higher_stats['mean']:.2f}) "
                f"had significantly higher scores than '{lower_var}' "
                f"(Mdn = {lower_stats['median']:.2f}, M = {lower_stats['mean']:.2f})."
            )
        
        return interpretation.strip()
        
    def _interpret_kruskal_wallis(self, is_sig, groups, effect_size_interp, es, p, alpha, h_stat, df, group_col, value_col):
        p_val_text = f"p < .001" if p < 0.001 else f"p = {p:.3f}"
        
        interpretation = (
            f"A Kruskal-Wallis H test was conducted to determine if there was a statistically significant "
            f"difference in '{value_col}' across groups of '{group_col}'.\n\n"
            f"The results indicated a {'statistically significant' if is_sig else 'not statistically significant'} "
            f"difference, H({df}) = {h_stat:.2f}, {p_val_text}.\n\n"
            f"The effect size was {effect_size_interp['text'].lower()} (ε² = {es:.3f})."
        )
        
        if is_sig:
            interpretation += (
                f"\n\nPost-hoc pairwise comparisons would be needed to determine which specific groups differ significantly."
            )
        
        return interpretation.strip()

    def _interpret_friedman(self, is_sig, var, effect_size_interp, W, p, alpha, stat, df):
        p_val_text = f"p < .001" if p < 0.001 else f"p = {p:.3f}"
        
        interpretation = (
            f"A Friedman test was conducted to determine if there was a statistically significant "
            f"difference across the repeated measures: {', '.join(var)}.\n\n"
            f"The results indicated a {'statistically significant' if is_sig else 'not statistically significant'} "
            f"difference, χ²({df}) = {stat:.2f}, {p_val_text}.\n\n"
            f"Kendall's W = {W:.3f}, indicating {effect_size_interp['text'].lower()} among the rankings."
        )
        
        if is_sig:
            interpretation += (
                f"\n\nPost-hoc pairwise comparisons would be needed to determine which specific conditions differ significantly."
            )
        
        return interpretation.strip()

    def plot_results(self, test_type):
        result = self.results.get(test_type)
        if not result:
            # Check if it's nested
            if self.results and self.results.get('results'):
                result = self.results.get('results')
            else:
                raise ValueError(f"No results found for test type: {test_type}")
        
        if test_type == 'mcnemar':
            table = pd.DataFrame(result['contingency_table'])
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Enhanced heatmap design
            sns.heatmap(table, annot=True, fmt='d', cmap='crest', 
                       cbar_kws={'label': 'Count'},
                       linewidths=1, linecolor='white',
                       square=True, ax=ax)
            ax.set_title(f"McNemar's Test Contingency Table", fontsize=14, fontweight='bold', pad=20)
            
            # Add variable names as labels
            vars = result.get('variables', ['Variable 1', 'Variable 2'])
            ax.set_xlabel(f'{vars[1]} Categories', fontsize=12)
            ax.set_ylabel(f'{vars[0]} Categories', fontsize=12)
            
            # Add test result annotation
            p_val = result['p_value']
            p_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            sig_text = "Significant" if result['is_significant'] else "Not Significant"
            ax.text(0.5, -0.15, f"χ² = {result['statistic']:.2f}, {p_text} ({sig_text})", 
                   transform=ax.transAxes, ha='center', fontsize=11)
            
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            if test_type == 'mann_whitney':
                groups = result['groups']
                group_col = result['group_col']
                value_col = result['value_col']
                
                # Box plot
                sns.boxplot(x=group_col, y=value_col, data=self.data, 
                          palette='crest', ax=axes[0, 0])
                axes[0, 0].set_title('Distribution by Group', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel(group_col, fontsize=11)
                axes[0, 0].set_ylabel(value_col, fontsize=11)
                
                # Mean ranks bar plot
                mean_ranks = [result['descriptive_stats'][g]['mean_rank'] for g in groups]
                colors = sns.color_palette('crest', n_colors=2)
                axes[0, 1].bar(groups, mean_ranks, color=colors, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Mean Ranks Comparison', fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel('Mean Rank', fontsize=11)
                axes[0, 1].set_xlabel(group_col, fontsize=11)
                
                # Violin plot for distribution comparison
                sns.violinplot(x=group_col, y=value_col, data=self.data, 
                             palette='crest', ax=axes[1, 0])
                axes[1, 0].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel(group_col, fontsize=11)
                axes[1, 0].set_ylabel(value_col, fontsize=11)
                
                # Summary statistics table
                axes[1, 1].axis('tight')
                axes[1, 1].axis('off')
                
                table_data = []
                for g in groups:
                    stats = result['descriptive_stats'][g]
                    table_data.append([g, stats['n'], f"{stats['median']:.2f}", 
                                     f"{stats['mean']:.2f}", f"{stats['std']:.2f}",
                                     f"{stats['mean_rank']:.2f}"])
                
                table = axes[1, 1].table(cellText=table_data,
                                       colLabels=['Group', 'n', 'Median', 'Mean', 'SD', 'Mean Rank'],
                                       cellLoc='center',
                                       loc='center',
                                       colWidths=[0.15, 0.1, 0.15, 0.15, 0.15, 0.15])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                # Style the header
                for (i, j), cell in table.get_celld().items():
                    if i == 0:
                        cell.set_facecolor('#E8F4F8')
                        cell.set_text_props(weight='bold')

            elif test_type == 'wilcoxon':
                vars = result['variables']
                
                # Paired box plots
                sns.boxplot(data=self.data[vars], palette='crest', ax=axes[0, 0])
                axes[0, 0].set_title('Paired Variables Distribution', fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel('Values', fontsize=11)
                
                # Distribution of differences
                differences = self.data[vars[0]] - self.data[vars[1]]
                sns.histplot(differences.dropna(), kde=True, color='#5B9BD5', ax=axes[0, 1])
                axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[0, 1].set_title('Distribution of Differences', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel(f'{vars[0]} - {vars[1]}', fontsize=11)
                axes[0, 1].set_ylabel('Frequency', fontsize=11)
                
                # Scatter plot of paired observations
                axes[1, 0].scatter(self.data[vars[0]], self.data[vars[1]], alpha=0.6, color='#5B9BD5')
                min_val = min(self.data[vars[0]].min(), self.data[vars[1]].min())
                max_val = max(self.data[vars[0]].max(), self.data[vars[1]].max())
                axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                axes[1, 0].set_title('Paired Observations', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel(vars[0], fontsize=11)
                axes[1, 0].set_ylabel(vars[1], fontsize=11)
                
                # Summary statistics table
                axes[1, 1].axis('tight')
                axes[1, 1].axis('off')
                
                table_data = []
                for v in vars:
                    stats = result['descriptive_stats'][v]
                    table_data.append([v, stats['n'], f"{stats['median']:.2f}", 
                                     f"{stats['mean']:.2f}", f"{stats['std']:.2f}"])
                
                table = axes[1, 1].table(cellText=table_data,
                                       colLabels=['Variable', 'n', 'Median', 'Mean', 'SD'],
                                       cellLoc='center',
                                       loc='center',
                                       colWidths=[0.2, 0.1, 0.15, 0.15, 0.15])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                for (i, j), cell in table.get_celld().items():
                    if i == 0:
                        cell.set_facecolor('#E8F4F8')
                        cell.set_text_props(weight='bold')
            
            elif test_type == 'kruskal_wallis':
                group_col = result['group_col']
                value_col = result['value_col']
                
                # Box plot
                sns.boxplot(x=group_col, y=value_col, data=self.data, 
                          palette='crest', ax=axes[0, 0])
                axes[0, 0].set_title('Distribution by Group', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel(group_col, fontsize=11)
                axes[0, 0].set_ylabel(value_col, fontsize=11)
                
                # Violin plot
                sns.violinplot(x=group_col, y=value_col, data=self.data, 
                             palette='crest', ax=axes[0, 1])
                axes[0, 1].set_title('Violin Plot by Group', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel(group_col, fontsize=11)
                axes[0, 1].set_ylabel(value_col, fontsize=11)
                
                # Mean ranks
                groups = list(result['descriptive_stats'].keys())
                mean_ranks = []
                for g in groups:
                    group_data = self.data[self.data[group_col] == g][value_col].dropna()
                    all_data = self.data[value_col].dropna()
                    ranks = rankdata(all_data)
                    group_indices = self.data[value_col].notna() & (self.data[group_col] == g)
                    mean_rank = ranks[group_indices.values[:len(ranks)]].mean() if any(group_indices) else 0
                    mean_ranks.append(mean_rank)
                
                colors = sns.color_palette('crest', n_colors=len(groups))
                axes[1, 0].bar(range(len(groups)), mean_ranks, color=colors, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Mean Ranks by Group', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel(group_col, fontsize=11)
                axes[1, 0].set_ylabel('Mean Rank', fontsize=11)
                axes[1, 0].set_xticks(range(len(groups)))
                axes[1, 0].set_xticklabels(groups, rotation=45 if len(groups) > 5 else 0)
                
                # Summary statistics table
                axes[1, 1].axis('tight')
                axes[1, 1].axis('off')
                
                table_data = []
                for g in groups[:10]:  # Limit to 10 groups for readability
                    stats = result['descriptive_stats'][g]
                    table_data.append([str(g)[:15], stats['n'], f"{stats['median']:.2f}", 
                                     f"{stats['mean']:.2f}", f"{stats['std']:.2f}"])
                
                table = axes[1, 1].table(cellText=table_data,
                                       colLabels=['Group', 'n', 'Median', 'Mean', 'SD'],
                                       cellLoc='center',
                                       loc='center',
                                       colWidths=[0.25, 0.1, 0.15, 0.15, 0.15])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                for (i, j), cell in table.get_celld().items():
                    if i == 0:
                        cell.set_facecolor('#E8F4F8')
                        cell.set_text_props(weight='bold')
                
            elif test_type == 'friedman':
                vars = result['variables']
                
                # Box plots for all conditions
                melted_data = self.data[vars].melt(var_name='Condition', value_name='Score')
                sns.boxplot(x='Condition', y='Score', data=melted_data, 
                          palette='crest', ax=axes[0, 0])
                axes[0, 0].set_title('Distribution by Condition', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('Condition', fontsize=11)
                axes[0, 0].set_ylabel('Score', fontsize=11)
                if len(vars) > 5:
                    axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Violin plot
                sns.violinplot(x='Condition', y='Score', data=melted_data, 
                             palette='crest', ax=axes[0, 1])
                axes[0, 1].set_title('Violin Plot by Condition', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Condition', fontsize=11)
                axes[0, 1].set_ylabel('Score', fontsize=11)
                if len(vars) > 5:
                    axes[0, 1].tick_params(axis='x', rotation=45)
                
                # Mean ranks
                mean_ranks = [self.data[v].rank().mean() for v in vars]
                colors = sns.color_palette('crest', n_colors=len(vars))
                axes[1, 0].bar(range(len(vars)), mean_ranks, color=colors, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Mean Ranks by Condition', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel('Condition', fontsize=11)
                axes[1, 0].set_ylabel('Mean Rank', fontsize=11)
                axes[1, 0].set_xticks(range(len(vars)))
                axes[1, 0].set_xticklabels(vars, rotation=45 if len(vars) > 5 else 0)
                
                # Summary statistics table
                axes[1, 1].axis('tight')
                axes[1, 1].axis('off')
                
                table_data = []
                for v in vars[:10]:  # Limit for readability
                    stats = result['descriptive_stats'][v]
                    table_data.append([v[:15], stats['n'], f"{stats['median']:.2f}", 
                                     f"{stats['mean']:.2f}", f"{stats['std']:.2f}"])
                
                table = axes[1, 1].table(cellText=table_data,
                                       colLabels=['Condition', 'n', 'Median', 'Mean', 'SD'],
                                       cellLoc='center',
                                       loc='center',
                                       colWidths=[0.25, 0.1, 0.15, 0.15, 0.15])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                for (i, j), cell in table.get_celld().items():
                    if i == 0:
                        cell.set_facecolor('#E8F4F8')
                        cell.set_text_props(weight='bold')

            # Add overall title with test results
            p_val = result['p_value']
            p_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            sig_text = "Significant" if result['is_significant'] else "Not Significant"
            fig.suptitle(f"{result['test_type']} Results ({p_text}, {sig_text})", 
                        fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        test_type = payload.get('testType')
        params = payload.get('params')

        if data is None or not test_type or not params:
            raise ValueError("Missing data, testType, or params")

        tester = NonParametricTests(data)
        result = {}
        
        if test_type == 'mann_whitney':
            result = tester.mann_whitney_u_test(**params)
        elif test_type == 'wilcoxon':
            result = tester.wilcoxon_signed_rank_test(**params)
        elif test_type == 'kruskal_wallis':
            result = tester.kruskal_wallis_test(**params)
        elif test_type == 'friedman':
            result = tester.friedman_test(**params)
        elif test_type == 'mcnemar':
            result = tester.mcnemar_test(**params)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        tester.results[test_type] = result
        plot_image = tester.plot_results(test_type)
        
        response = { 'results': result, 'plot': plot_image }
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
    