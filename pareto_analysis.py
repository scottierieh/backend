import sys
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def calculate_insights(freq_df, category_variable, value_label):
    """Calculate insights from Pareto analysis"""
    insights = {}
    
    # Find items contributing to 80%
    vital_few = freq_df[freq_df['Cumulative Percentage'] <= 80]
    if len(vital_few) == 0:
        vital_few = freq_df.head(1)
    
    vital_count = len(vital_few)
    total_count = len(freq_df)
    vital_percentage = (vital_count / total_count) * 100
    
    insights['vital_few_count'] = vital_count
    insights['total_categories'] = total_count
    insights['vital_few_percentage'] = round(vital_percentage, 1)
    insights['vital_few_contribution'] = round(vital_few['Percentage'].sum(), 1)
    
    # Top contributors
    insights['top_3_items'] = vital_few.head(3)['Value'].tolist()
    insights['top_3_contribution'] = round(freq_df.head(3)['Percentage'].sum(), 1)
    
    # Calculate concentration ratio (Gini coefficient approximation)
    cumsum = freq_df['Percentage'].cumsum()
    n = len(freq_df)
    concentration = 1 - (2 / n) * cumsum.sum() / 100
    insights['concentration_index'] = round(concentration, 3)
    
    # Interpretation
    if vital_percentage <= 25:
        insights['interpretation'] = 'Strong Pareto effect: Very few categories drive most impact'
        insights['severity'] = 'high'
    elif vital_percentage <= 40:
        insights['interpretation'] = 'Moderate Pareto effect: Focus on key categories recommended'
        insights['severity'] = 'medium'
    else:
        insights['interpretation'] = 'Weak Pareto effect: Impact is more evenly distributed'
        insights['severity'] = 'low'
    
    return insights

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        category_variable = payload.get('variable')
        value_variable = payload.get('valueVariable')  # Optional
        filter_top_n = payload.get('filterTopN')  # Optional

        if not data or not category_variable:
            raise ValueError("Missing 'data' or 'variable'")

        df = pd.DataFrame(data)
        
        if category_variable not in df.columns:
            raise ValueError(f"Variable '{category_variable}' not found in data.")

        # Remove null values from category variable
        df = df[df[category_variable].notna()].copy()
        
        if len(df) == 0:
            raise ValueError(f"No valid data after removing null values from '{category_variable}'")

        # Calculate aggregation
        if value_variable and value_variable in df.columns:
            # Pareto with values (e.g., customer sales, defect costs)
            df[value_variable] = pd.to_numeric(df[value_variable], errors='coerce')
            df = df[df[value_variable].notna()]
            
            if len(df) == 0:
                raise ValueError(f"No valid numeric data in '{value_variable}'")
            
            freq_df = df.groupby(category_variable)[value_variable].sum().reset_index()
            freq_df.columns = ['Value', 'Amount']
            value_label = value_variable
        else:
            # Frequency-based Pareto (e.g., defect types count)
            freq_df = df[category_variable].value_counts().reset_index()
            freq_df.columns = ['Value', 'Amount']
            value_label = 'Frequency'
        
        # Sort by amount/frequency descending
        freq_df = freq_df.sort_values(by='Amount', ascending=False).reset_index(drop=True)
        
        # Apply top N filter if specified
        if filter_top_n and filter_top_n > 0:
            original_count = len(freq_df)
            freq_df = freq_df.head(filter_top_n)
            filtered_info = {
                'applied': True,
                'showing': len(freq_df),
                'total': original_count
            }
        else:
            filtered_info = {'applied': False}
        
        # Calculate percentage and cumulative percentage
        total = freq_df['Amount'].sum()
        freq_df['Percentage'] = (freq_df['Amount'] / total) * 100
        freq_df['Cumulative Percentage'] = freq_df['Percentage'].cumsum()

        # Limit to top categories if too many (prevents overcrowding)
        max_categories = 25
        has_others = False
        if len(freq_df) > max_categories:
            top_df = freq_df.head(max_categories - 1).copy()
            others_amount = freq_df.iloc[max_categories - 1:]['Amount'].sum()
            others_pct = (others_amount / total) * 100
            
            others_row = pd.DataFrame({
                'Value': ['Others'],
                'Amount': [others_amount],
                'Percentage': [others_pct],
                'Cumulative Percentage': [100.0]
            })
            freq_df = pd.concat([top_df, others_row], ignore_index=True)
            has_others = True

        # Calculate insights
        insights = calculate_insights(freq_df[freq_df['Value'] != 'Others'] if has_others else freq_df, 
                                     category_variable, value_label)

        # Find 80% cutoff for coloring
        cutoff_80 = freq_df[freq_df['Cumulative Percentage'] >= 80].index
        vital_few_indices = set(range(cutoff_80[0] + 1)) if len(cutoff_80) > 0 else set()

        # Prepare data for plotting
        categories = [str(val)[:40] + '...' if len(str(val)) > 40 else str(val) 
                     for val in freq_df['Value']]
        amounts = freq_df['Amount'].tolist()
        percentages = freq_df['Percentage'].tolist()
        cumulative = freq_df['Cumulative Percentage'].tolist()
        
        # Colors: green for vital few, blue for trivial many
        colors = ['#2E7D32' if i in vital_few_indices else '#42A5F5' 
                  for i in range(len(freq_df))]

        # --- Create Interactive Plotly Chart ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar chart for amounts
        fig.add_trace(
            go.Bar(
                x=categories,
                y=amounts,
                name=value_label,
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                hovertemplate=(
                    '<b>%{x}</b><br>' +
                    f'{value_label}: %{{y:,.0f}}<br>' +
                    'Percentage: %{customdata[0]:.1f}%<br>' +
                    'Cumulative: %{customdata[1]:.1f}%<br>' +
                    'Priority: %{customdata[2]}<br>' +
                    '<extra></extra>'
                ),
                customdata=[[p, c, 'High' if i in vital_few_indices else 'Low'] 
                           for i, (p, c) in enumerate(zip(percentages, cumulative))],
                yaxis='y1'
            ),
            secondary_y=False
        )

        # Line chart for cumulative percentage
        fig.add_trace(
            go.Scatter(
                x=categories,
                y=cumulative,
                name='Cumulative %',
                mode='lines+markers',
                line=dict(color='#D32F2F', width=3),
                marker=dict(
                    size=8,
                    color='white',
                    line=dict(color='#D32F2F', width=2.5)
                ),
                hovertemplate=(
                    '<b>%{x}</b><br>' +
                    'Cumulative: %{y:.1f}%<br>' +
                    '<extra></extra>'
                ),
                yaxis='y2'
            ),
            secondary_y=True
        )

        # Add 80% reference line
        fig.add_hline(
            y=80,
            line=dict(color='#FF6F00', dash='dash', width=2.5),
            annotation_text='80% Threshold',
            annotation_position='right',
            annotation=dict(font=dict(size=12, color='#FF6F00')),
            secondary_y=True
        )

        # Add shaded region below 80%
        fig.add_hrect(
            y0=0, y1=80,
            fillcolor='green',
            opacity=0.05,
            layer='below',
            line_width=0,
            secondary_y=True
        )

        # Add vertical line at 80% cutoff
        if len(cutoff_80) > 0:
            cutoff_idx = cutoff_80[0]
            fig.add_vline(
                x=cutoff_idx,
                line=dict(color='#FF6F00', dash='dot', width=2),
                opacity=0.6,
                annotation_text=f'{insights["vital_few_count"]} items = {insights["vital_few_contribution"]}%',
                annotation_position='top',
                annotation=dict(
                    font=dict(size=11, color='#FF6F00'),
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='#FF6F00',
                    borderwidth=2,
                    borderpad=4
                )
            )

        # Update layout
        title = f'Pareto Chart: {category_variable}'
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='#1a1a1a'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(text=category_variable, font=dict(size=13, color='#1a1a1a')),
                tickangle=-45,
                tickfont=dict(size=10),
                showgrid=False
            ),
            yaxis=dict(
                title=dict(text=value_label, font=dict(size=13, color='#1565C0')),
                tickfont=dict(color='#1565C0'),
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1
            ),
            yaxis2=dict(
                title=dict(text='Cumulative Percentage (%)', font=dict(size=13, color='#D32F2F')),
                tickfont=dict(color='#D32F2F'),
                range=[0, 105],
                showgrid=False
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='monospace'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(size=11)
            ),
            plot_bgcolor='rgba(250,250,250,0.5)',
            paper_bgcolor='white',
            height=600,
            margin=dict(l=80, r=80, t=100, b=120)
        )

        # Add custom legend items for priority colors
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name='Vital Few (≤80%)',
                marker=dict(color='#2E7D32'),
                showlegend=True,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name='Trivial Many',
                marker=dict(color='#42A5F5'),
                showlegend=True,
                hoverinfo='skip'
            )
        )

        # Convert to JSON
        plot_json = fig.to_json()

        # Prepare table data with priority flag
        table_data = freq_df.copy()
        table_data['Priority'] = table_data.index.map(
            lambda i: 'High' if i in vital_few_indices and table_data.loc[i, 'Value'] != 'Others' 
            else 'Low'
        )
        table_data['Amount'] = table_data['Amount'].apply(_to_native_type)
        table_data['Percentage'] = table_data['Percentage'].apply(_to_native_type)
        table_data['Cumulative Percentage'] = table_data['Cumulative Percentage'].apply(_to_native_type)

        response = {
            'table': table_data.to_dict('records'),
            'plot': plot_json,
            'insights': insights,
            'filter_info': filtered_info
        }
        
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
    