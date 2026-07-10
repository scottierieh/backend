

import sys
import json
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings('ignore')
pio.templates.default = "plotly_white"

# New color palette
PALETTE = ['#a67b70', '#b5a888', '#c4956a', '#7a9471', '#8ba3a3', '#6b7565', '#d4c4a8', '#9a8471', '#a8b5a3']

def to_json(fig):
    """Converts a Plotly figure to a JSON string."""
    return pio.to_json(fig)

def create_bar_chart(data, x, y, title, xlabel, ylabel, orientation='v'):
    fig = go.Figure(go.Bar(
        x=data[x] if orientation == 'v' else data[y], 
        y=data[y] if orientation == 'v' else data[x], 
        orientation=orientation,
        marker_color=PALETTE[0]
    ))
    fig.update_layout(title_text=title, xaxis_title=xlabel, yaxis_title=ylabel, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_funnel_chart(data):
    fig = go.Figure(go.Funnel(
        y=data['Stage'],
        x=data['Count'],
        textinfo="value+percent initial",
        marker={"color": [PALETTE[0], PALETTE[1]]}
    ))
    fig.update_layout(title_text='Simplified Conversion Funnel', margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_box_plot(df, x_col, y_col, title):
    fig = go.Figure()
    for i, category in enumerate(df[x_col].unique()):
        fig.add_trace(go.Box(y=df[df[x_col] == category][y_col], name=category, marker_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(title_text=title, yaxis_title=y_col)
    return fig

def create_treemap(df, path, values, title):
     fig = go.Figure(go.Treemap(
        labels=df[path[0]],
        parents=[""] * len(df),
        values=df[values],
        textinfo="label+value+percent root",
        marker_colors=PALETTE
    ))
     fig.update_layout(title_text=title, margin=dict(t=50, l=25, r=25, b=25))
     return fig
     
def create_pie_chart(df, labels_col, values_col, title, hole=0):
    counts = df[labels_col].value_counts()
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=hole, marker_colors=PALETTE)])
    fig.update_layout(title_text=title)
    return fig

def generate_traffic_interpretation(df, config):
    interp = "### Traffic & ROI Insights\n\n"
    # Campaign ROI
    try:
        campaign_col, cost_col, revenue_col = config.get('campaignCol'), config.get('costCol'), config.get('revenueCol')
        if all(c and c in df.columns for c in [campaign_col, cost_col, revenue_col]):
            campaign_stats = df.groupby(campaign_col).agg(
                total_cost=(cost_col, 'sum'),
                total_revenue=(revenue_col, 'sum')
            ).reset_index()
            campaign_stats['roi'] = np.where(campaign_stats['total_cost'] > 0, ((campaign_stats['total_revenue'] - campaign_stats['total_cost']) / campaign_stats['total_cost']) * 100, 0)
            best_roi = campaign_stats.sort_values('roi', ascending=False).iloc[0]
            interp += f"- **Top Performer:** The '{best_roi[campaign_col]}' campaign yields the highest ROI at **{best_roi['roi']:.1f}%**.\n"
    except Exception: pass
    
    # Channel Performance
    try:
        source_col, medium_col, conversion_col = config.get('sourceCol'), config.get('mediumCol'), config.get('conversionCol')
        if all(c and c in df.columns for c in [source_col, medium_col, conversion_col]):
            df['channel'] = df[source_col].astype(str) + ' / ' + df[medium_col].astype(str)
            conversion_rate = df.groupby('channel')[conversion_col].mean().sort_values(ascending=False) * 100
            best_channel = conversion_rate.index[0]
            interp += f"- **Most Effective Channel:** The '{best_channel}' channel shows the best conversion rate (**{conversion_rate.iloc[0]:.1f}%**).\n"
    except Exception: pass

    interp += "\n**Recommendation:** Allocate more budget to high-performing channels and campaigns. Investigate underperforming campaigns to optimize or discontinue them."
    return interp


def generate_ecommerce_interpretation(df, config):
    interp = "### E-commerce Insights\n\n"
    # Cohort Analysis
    try:
        user_id_col, cohort_date_col, date_col = config.get('userIdCol'), config.get('cohortDateCol'), config.get('dateCol')
        if all(c and c in df.columns for c in [user_id_col, cohort_date_col, date_col]):
             df_cohort = df[[user_id_col, cohort_date_col, date_col]].copy().dropna()
             df_cohort['order_month'] = df_cohort[date_col].dt.to_period('M')
             df_cohort['cohort'] = df_cohort.groupby(user_id_col)[cohort_date_col].transform('min').dt.to_period('M')
             df_cohort_agg = df_cohort.groupby(['cohort', 'order_month']).agg(n_customers=(user_id_col, 'nunique')).reset_index(drop=False)
             df_cohort_agg['period_number'] = (df_cohort_agg.order_month - df_cohort_agg.cohort).apply(lambda x: x.n)
             if len(df_cohort_agg) > 1 and 'period_number' in df_cohort_agg.columns and df_cohort_agg['period_number'].max() > 0:
                 interp += "- **Retention:** The cohort analysis shows how customer retention evolves over time. Focus on improving Month 1 and Month 2 retention to boost long-term value.\n"
    except Exception: pass

    # Basket Analysis
    try:
        item_category_col, item_brand_col = config.get('itemCategoryCol'), config.get('itemBrandCol')
        if all(c and c in df.columns for c in [item_category_col, item_brand_col]):
            interp += "- **Product Associations:** The heatmap of item category vs. brand reveals co-purchase patterns. Consider creating product bundles or cross-promotions based on strong associations.\n"
    except Exception: pass
    
    interp += "\n**Recommendation:** Use retention data to time re-engagement campaigns. Leverage product associations for targeted marketing and upselling opportunities."
    return interp

def generate_segmentation_interpretation(df, config):
    interp = "### Customer Segmentation Insights\n\n"
    # Age & Gender
    try:
        age_col, gender_col = config.get('ageGroupCol'), config.get('genderCol')
        if all(c and c in df.columns for c in [age_col, gender_col]):
            segment_counts = df.groupby([age_col, gender_col]).size().reset_index(name='count').sort_values('count', ascending=False)
            top_segment = segment_counts.iloc[0]
            interp += f"- **Dominant Segment:** Your largest customer segment is **{top_segment[age_col]} {top_segment[gender_col]}s**. Tailor marketing messages and product offerings to this group.\n"
    except Exception: pass
    
    # LTV by Membership
    try:
        ltv_col, membership_col = config.get('ltvCol'), config.get('membershipCol')
        if all(c and c in df.columns for c in [ltv_col, membership_col]):
            ltv_means = df.groupby(membership_col)[ltv_col].mean().sort_values(ascending=False)
            if len(ltv_means) > 0:
                highest_ltv_group = ltv_means.index[0]
                interp += f"- **High-Value Customers:** '{highest_ltv_group}' members have the highest average Lifetime Value (LTV). Focus on retaining these customers and upselling lower-tier members.\n"
    except Exception: pass
    
    interp += "\n**Recommendation:** Develop targeted campaigns for your primary demographic. Create loyalty programs to nurture high-LTV customers and encourage others to upgrade."
    return interp

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        config = payload.get('config', {})

        if not data:
            raise ValueError("No data provided.")

        df = pd.DataFrame(data)
        plots = {}
        interpretations = {}
        
        # --- Convert types ---
        numeric_cols_keys = [
            'revenueCol', 'costCol', 'ltvCol', 'conversionCol', 'priceCol', 'quantityCol'
        ]
        date_cols_keys = ['dateCol', 'cohortDateCol']
        
        for key in numeric_cols_keys:
            col_name = config.get(key)
            if col_name and col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        for key in date_cols_keys:
            col_name = config.get(key)
            if col_name and col_name in df.columns:
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

        # --- Generate Plots & Interpretations ---
        try:
            interpretations['traffic'] = generate_traffic_interpretation(df, config)
            # Traffic Analysis
            source_col, medium_col, conversion_col = config.get('sourceCol'), config.get('mediumCol'), config.get('conversionCol')
            if all(c and c in df.columns for c in [source_col, medium_col, conversion_col]):
                df['channel'] = df[source_col].astype(str) + ' / ' + df[medium_col].astype(str)
                conversion_rate = df.groupby('channel')[conversion_col].mean().sort_values(ascending=False) * 100
                fig = create_bar_chart(conversion_rate.reset_index(), x='channel', y=conversion_col, title='Conversion Rate by Channel', xlabel='Channel', ylabel='Conversion Rate (%)')
                plots['channelPerformance'] = to_json(fig)
            
            campaign_col, cost_col, revenue_col = config.get('campaignCol'), config.get('costCol'), config.get('revenueCol')
            if all(c and c in df.columns for c in [campaign_col, cost_col, revenue_col]):
                campaign_stats = df.groupby(campaign_col).agg(total_cost=(cost_col, 'sum'), total_revenue=(revenue_col, 'sum')).reset_index()
                campaign_stats['roi'] = np.where(campaign_stats['total_cost'] > 0, ((campaign_stats['total_revenue'] - campaign_stats['total_cost']) / campaign_stats['total_cost']) * 100, 0)
                fig = create_bar_chart(campaign_stats.sort_values('roi'), x='roi', y=campaign_col, title='Campaign ROI (%)', xlabel='ROI (%)', ylabel='Campaign', orientation='h')
                plots['campaignRoi'] = to_json(fig)

            if conversion_col and conversion_col in df.columns:
                funnel_df = pd.DataFrame({'Stage': ['Total Sessions', 'Converted'], 'Count': [len(df), len(df[df[conversion_col] == 1])]})
                fig = create_funnel_chart(funnel_df)
                plots['funnelAnalysis'] = to_json(fig)
            
            revenue_col = config.get('revenueCol')
            if all(c and c in df.columns for c in [source_col, conversion_col, revenue_col]):
                converted_df = df[df[conversion_col] == 1].copy()
                if not converted_df.empty:
                    attribution = converted_df.groupby(source_col)[revenue_col].sum().sort_values().reset_index()
                    fig = create_bar_chart(attribution, x=revenue_col, y=source_col, title='Attributed Revenue (Last Touch)', xlabel='Attributed Revenue', ylabel='Source', orientation='h')
                    plots['attributionModeling'] = to_json(fig)
        except Exception as e:
            plots['traffic_error'] = str(e)

        try:
            interpretations['ecommerce'] = generate_ecommerce_interpretation(df, config)
            # E-commerce Analysis
            user_id_col, cohort_date_col, date_col = config.get('userIdCol'), config.get('cohortDateCol'), config.get('dateCol')
            if all(c and c in df.columns for c in [user_id_col, cohort_date_col, date_col]):
                df_cohort = df[[user_id_col, cohort_date_col, date_col]].copy().dropna()
                df_cohort['order_month'] = df_cohort[date_col].dt.to_period('M')
                df_cohort['cohort'] = df_cohort.groupby(user_id_col)[cohort_date_col].transform('min').dt.to_period('M')
                df_cohort_agg = df_cohort.groupby(['cohort', 'order_month']).agg(n_customers=(user_id_col, 'nunique')).reset_index(drop=False)
                df_cohort_agg['period_number'] = (df_cohort_agg.order_month - df_cohort_agg.cohort).apply(lambda x: x.n)
                cohort_pivot = df_cohort_agg.pivot_table(index='cohort', columns='period_number', values='n_customers')
                retention_matrix = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0) * 100
                fig = go.Figure(data=go.Heatmap(z=retention_matrix.values, x=retention_matrix.columns.astype(str), y=retention_matrix.index.astype(str), colorscale=[[0, PALETTE[1]], [1, PALETTE[0]]], text=np.around(retention_matrix.values, 2), texttemplate="%{text}%"))
                fig.update_layout(title='Monthly Cohort Retention Rate (%)', xaxis_title='Months Since First Purchase', yaxis_title='Cohort')
                plots['cohortAnalysis'] = to_json(fig)
            
            item_category_col, item_brand_col = config.get('itemCategoryCol'), config.get('itemBrandCol')
            if all(c and c in df.columns for c in [item_category_col, item_brand_col]):
                crosstab = pd.crosstab(df[item_category_col], df[item_brand_col])
                fig = go.Figure(data=go.Heatmap(z=crosstab.values, x=crosstab.columns, y=crosstab.index, colorscale=[[0, '#d4c4a8'],[1, '#7a9471']]))
                fig.update_layout(title='Item Category vs. Brand Co-occurrence')
                plots['basketAnalysis'] = to_json(fig)
            
            price_col, quantity_col = config.get('priceCol'), config.get('quantityCol')
            if all(c and c in df.columns for c in [price_col, quantity_col]):
                fig = go.Figure(data=go.Scatter(x=df[price_col], y=df[quantity_col], mode='markers', marker_color=PALETTE[4]))
                fig.update_layout(title='Price vs. Quantity (Price Elasticity)', xaxis_title='Price', yaxis_title='Quantity Sold')
                plots['priceElasticity'] = to_json(fig)
            
            coupon_col = config.get('couponUsedCol')
            if all(c and c in df.columns for c in [coupon_col, conversion_col]):
                coupon_effect = df.groupby(coupon_col)[conversion_col].mean().sort_values() * 100
                fig = create_bar_chart(coupon_effect.reset_index(), x=coupon_col, y=conversion_col, title='Coupon Usage vs. Conversion Rate', xlabel='Coupon Used', ylabel='Conversion Rate (%)')
                plots['couponEffectiveness'] = to_json(fig)
        except Exception as e:
            plots['ecommerce_error'] = str(e)
            
        try:
            interpretations['segmentation'] = generate_segmentation_interpretation(df, config)
            # Customer Segmentation
            age_col, gender_col = config.get('ageGroupCol'), config.get('genderCol')
            if all(c and c in df.columns for c in [age_col, gender_col]):
                segment_counts = df.groupby([age_col, gender_col]).size().reset_index(name='count')
                path_labels = segment_counts.apply(lambda row: f"{row[age_col]} - {row[gender_col]}", axis=1)
                fig = create_treemap(segment_counts.assign(path=path_labels), path=['path'], values='count', title='Customer Segments by Age & Gender')
                plots['segmentationTreemap'] = to_json(fig)
            
            country_col = config.get('countryCol')
            if country_col and country_col in df.columns:
                country_counts = df[country_col].value_counts().reset_index()
                fig = create_bar_chart(country_counts, x='count', y=country_col, title='Geographic Distribution of Users', xlabel='Number of Users', ylabel='Country', orientation='h')
                plots['geographicDistribution'] = to_json(fig)
            
            ltv_col, membership_col = config.get('ltvCol'), config.get('membershipCol')
            if all(c and c in df.columns for c in [ltv_col, membership_col]):
                fig = create_box_plot(df, x_col=membership_col, y_col=ltv_col, title='LTV by Membership Level')
                plots['ltvByMembership'] = to_json(fig)
            
            device_col = config.get('deviceCol')
            if device_col and device_col in df.columns:
                fig = create_pie_chart(df, labels_col=device_col, values_col=None, title='Sessions by Device', hole=0.4)
                plots['deviceUsage'] = to_json(fig)
        except Exception as e:
            plots['segmentation_error'] = str(e)
        
        response = {'plots': plots, 'interpretations': interpretations}
        print(json.dumps(response))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
