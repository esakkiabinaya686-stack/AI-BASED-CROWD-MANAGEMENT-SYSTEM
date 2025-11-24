# integrated_analytics_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Crowd Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid;
        margin: 0.5rem;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-content {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .alert-low { border-left-color: #00c853; background-color: #e8f5e8; }
    .alert-normal { border-left-color: #ffd600; background-color: #fffde7; }
    .alert-high { border-left-color: #ff9100; background-color: #fff3e0; }
    .alert-critical { border-left-color: #ff1744; background-color: #ffebee; }
    .live-badge {
        background: linear-gradient(45deg, #ff4444, #ff6b6b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .metric-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #2c3e50;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-subtitle {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin: 0;
    }
    .alarm-active {
        background: linear-gradient(45deg, #ff4444, #ff6b6b);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        animation: alarm-pulse 1s infinite;
    }
    .alarm-inactive {
        background: linear-gradient(45deg, #00c853, #4caf50);
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    @keyframes alarm-pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .hypothesis-result {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    .hypothesis-rejected {
        background-color: #ffebee;
        color: #d32f2f;
        border: 2px solid #d32f2f;
    }
    .hypothesis-accepted {
        background-color: #e8f5e8;
        color: #388e3c;
        border: 2px solid #388e3c;
    }
</style>
""", unsafe_allow_html=True)

class IntegratedCrowdAnalyticsDashboard:
    def __init__(self):
        self.data_file = "crowd_data.json"
        self.initialize_data_file()
    
    def initialize_data_file(self):
        """Initialize data file with default values"""
        default_data = {
            'current_count': 0,
            'alert_level': 'LOW',
            'total_alerts': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'people_history': [],
            'status_message': 'Waiting for data from detection system...',
            'frames_processed': 0,
            'average_crowd': 0.0,
            'maximum_crowd': 0,
            'alarm_active': False,
            'system_status': 'INACTIVE',
            'video_file': 'Not specified',
            'alert_thresholds': {
                'low': 10,
                'normal': 20,
                'high': 30
            }
        }
        
        try:
            if not os.path.exists(self.data_file):
                with open(self.data_file, 'w') as f:
                    json.dump(default_data, f, indent=4)
        except Exception as e:
            print(f"Error initializing data file: {e}")
    
    def load_data(self):
        """Load real-time data from JSON file"""
        default_data = {
            'current_count': 0,
            'alert_level': 'LOW',
            'total_alerts': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'people_history': [],
            'status_message': 'Waiting for data from detection system...',
            'frames_processed': 0,
            'average_crowd': 0.0,
            'maximum_crowd': 0,
            'alarm_active': False,
            'system_status': 'INACTIVE',
            'video_file': 'Not specified',
            'alert_thresholds': {
                'low': 10,
                'normal': 20,
                'high': 30
            }
        }
        
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for key in default_data:
                        if key not in data:
                            data[key] = default_data[key]
                    
                    # Update timestamp to current time for real-time feel
                    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    return data
            else:
                return default_data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return default_data
    
    def update_crowd_data(self, new_count):
        """Update crowd data with proper counting and alert logic"""
        data = self.load_data()
        
        # Update current count
        data['current_count'] = new_count
        
        # Update people history (keep last 100 entries)
        data['people_history'].append(new_count)
        if len(data['people_history']) > 100:
            data['people_history'] = data['people_history'][-100:]
        
        # Determine alert level
        thresholds = data.get('alert_thresholds', {'low': 10, 'normal': 20, 'high': 30})
        
        if new_count <= thresholds['low']:
            alert_level = 'LOW'
            alarm_active = False
        elif new_count <= thresholds['normal']:
            alert_level = 'NORMAL'
            alarm_active = False
        elif new_count <= thresholds['high']:
            alert_level = 'HIGH'
            alarm_active = True
        else:
            alert_level = 'CRITICAL'
            alarm_active = True
        
        # Update alert count if alarm state changed
        if alarm_active and not data['alarm_active']:
            data['total_alerts'] += 1
        
        # Update alarm status
        data['alarm_active'] = alarm_active
        data['alert_level'] = alert_level
        
        # Update statistics
        data['frames_processed'] = len(data['people_history'])
        if data['people_history']:
            data['average_crowd'] = np.mean(data['people_history'])
            data['maximum_crowd'] = np.max(data['people_history'])
        
        # Update system status
        data['system_status'] = 'ACTIVE' if data['frames_processed'] > 0 else 'INACTIVE'
        data['status_message'] = f'Monitoring {new_count} people - {alert_level} level'
        
        # Save updated data
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            st.error(f"Error saving data: {e}")
        
        return data
    
    def simulate_crowd_data(self):
        """Simulate realistic crowd data for testing"""
        import random
        
        # Load current data
        data = self.load_data()
        
        # Simulate realistic crowd patterns
        base_pattern = [5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 30, 25, 20, 15, 10, 8, 6]
        current_history_len = len(data['people_history'])
        
        if current_history_len >= len(base_pattern):
            # Add some randomness to existing pattern
            last_count = data['people_history'][-1] if data['people_history'] else 0
            new_count = max(0, last_count + random.randint(-3, 5))
            
            # Occasionally simulate crowd surges
            if random.random() < 0.1:  # 10% chance of surge
                new_count = random.randint(25, 40)
        else:
            # Follow the base pattern
            new_count = base_pattern[current_history_len % len(base_pattern)]
        
        return self.update_crowd_data(new_count)
    
    def calculate_statistics(self, data):
        """Calculate comprehensive statistics from real-time data"""
        if not data['people_history']:
            return {
                'mean': 0, 'median': 0, 'std_dev': 0, 'variance': 0, 
                'min': 0, 'max': 0, 'range': 0, 'q1': 0, 'q3': 0, 'iqr': 0,
                'skewness': 0, 'kurtosis': 0, 'alert_distribution': {'LOW': 0, 'NORMAL': 0, 'HIGH': 0, 'CRITICAL': 0},
                'total_frames': 0
            }
        
        counts = data['people_history']
        thresholds = data.get('alert_thresholds', {'low': 10, 'normal': 20, 'high': 30})
        
        stats_dict = {
            'mean': np.mean(counts),
            'median': np.median(counts),
            'std_dev': np.std(counts),
            'variance': np.var(counts),
            'min': np.min(counts),
            'max': np.max(counts),
            'range': np.max(counts) - np.min(counts),
            'q1': np.percentile(counts, 25),
            'q3': np.percentile(counts, 75),
            'iqr': np.percentile(counts, 75) - np.percentile(counts, 25),
            'skewness': scipy_stats.skew(counts),
            'kurtosis': scipy_stats.kurtosis(counts)
        }
        
        # Alert distribution
        alert_counts = {
            'LOW': len([x for x in counts if x <= thresholds['low']]),
            'NORMAL': len([x for x in counts if thresholds['low'] < x <= thresholds['normal']]),
            'HIGH': len([x for x in counts if thresholds['normal'] < x <= thresholds['high']]),
            'CRITICAL': len([x for x in counts if x > thresholds['high']])
        }
        
        stats_dict['alert_distribution'] = alert_counts
        stats_dict['total_frames'] = len(counts)
        
        return stats_dict
    
    def predictive_analysis(self, data):
        """Perform predictive modeling on real-time data"""
        if len(data['people_history']) < 10:
            return {
                'trend': 0,
                'future_predictions': [],
                'trend_line': []
            }
        
        counts = data['people_history']
        X = np.array(range(len(counts))).reshape(-1, 1)
        y = np.array(counts)
        
        # Linear Regression for trend analysis
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        trend = lr_model.coef_[0]
        
        # Random Forest for prediction
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Predict next 5 values
        future_X = np.array(range(len(counts), len(counts) + 5)).reshape(-1, 1)
        future_predictions = rf_model.predict(future_X)
        
        return {
            'trend': trend,
            'future_predictions': future_predictions.tolist(),
            'trend_line': lr_model.predict(X).tolist()
        }
    
    def risk_analysis(self, data):
        """Perform risk assessment on real-time data"""
        if not data['people_history']:
            return {
                'low_risk': 0,
                'medium_risk': 0,
                'high_risk': 0,
                'anomalies': [],
                'risk_score': 0
            }
        
        counts = data['people_history']
        
        # Z-score for anomaly detection
        if len(counts) > 1:
            z_scores = scipy_stats.zscore(counts)
            anomalies = np.where(np.abs(z_scores) > 2)[0].tolist()
        else:
            anomalies = []
        
        # Risk levels based on statistical thresholds
        mean_count = np.mean(counts) if counts else 0
        std_count = np.std(counts) if counts else 0
        
        high_risk_threshold = mean_count + 2 * std_count
        medium_risk_threshold = mean_count + std_count
        
        risk_assessment = {
            'low_risk': len([x for x in counts if x <= medium_risk_threshold]),
            'medium_risk': len([x for x in counts if medium_risk_threshold < x <= high_risk_threshold]),
            'high_risk': len([x for x in counts if x > high_risk_threshold]),
            'anomalies': anomalies,
            'risk_score': min(100, int((len(anomalies) / len(counts)) * 100)) if counts else 0
        }
        
        return risk_assessment
    
    def hypothesis_testing(self, data):
        """Perform T-test, Chi-square test, and ANOVA test on crowd data"""
        if len(data['people_history']) < 10:
            return {
                't_test': {'statistic': 0, 'p_value': 0, 'result': 'Insufficient data'},
                'chi_square_test': {'statistic': 0, 'p_value': 0, 'result': 'Insufficient data'},
                'anova_test': {'statistic': 0, 'p_value': 0, 'result': 'Insufficient data'}
            }
        
        counts = data['people_history']
        results = {}
        
        # 1. One-sample T-Test
        expected_mean = 15  # Expected average crowd size
        try:
            t_stat, t_p = scipy_stats.ttest_1samp(counts, expected_mean)
            results['t_test'] = {
                'test': 'One-Sample T-Test',
                'statistic': t_stat,
                'p_value': t_p,
                'result': 'Different from expected' if t_p < 0.05 else 'Not different from expected',
                'hypothesis': f'H0: Mean crowd size = {expected_mean}'
            }
        except Exception as e:
            results['t_test'] = {
                'test': 'One-Sample T-Test',
                'statistic': 0,
                'p_value': 1,
                'result': 'Test failed',
                'hypothesis': f'H0: Mean crowd size = {expected_mean}'
            }
        
        # 2. Chi-Square Goodness of Fit Test
        try:
            # Create observed frequencies for different crowd ranges
            observed = [
                len([x for x in counts if x <= 10]),   # Low crowd
                len([x for x in counts if 10 < x <= 20]),  # Normal crowd
                len([x for x in counts if x > 20])     # High crowd
            ]
            
            # Expected frequencies (equal distribution)
            expected = [len(counts)/3] * 3
            
            chi2_stat, chi2_p = scipy_stats.chisquare(observed, expected)
            results['chi_square_test'] = {
                'test': 'Chi-Square Goodness of Fit Test',
                'statistic': chi2_stat,
                'p_value': chi2_p,
                'result': 'Distribution differs from expected' if chi2_p < 0.05 else 'Distribution as expected',
                'hypothesis': 'H0: Crowd distribution follows expected pattern'
            }
        except Exception as e:
            results['chi_square_test'] = {
                'test': 'Chi-Square Goodness of Fit Test',
                'statistic': 0,
                'p_value': 1,
                'result': 'Test failed',
                'hypothesis': 'H0: Crowd distribution follows expected pattern'
            }
        
        # 3. ANOVA Test (comparing first half vs second half of data)
        try:
            if len(counts) >= 20:
                # Split data into two groups
                split_point = len(counts) // 2
                group1 = counts[:split_point]
                group2 = counts[split_point:]
                
                # Perform one-way ANOVA
                f_stat, f_p = scipy_stats.f_oneway(group1, group2)
                results['anova_test'] = {
                    'test': 'One-Way ANOVA Test',
                    'statistic': f_stat,
                    'p_value': f_p,
                    'result': 'Significant difference between groups' if f_p < 0.05 else 'No significant difference',
                    'hypothesis': 'H0: No difference between first and second half of data'
                }
            else:
                results['anova_test'] = {
                    'test': 'One-Way ANOVA Test',
                    'statistic': 0,
                    'p_value': 1,
                    'result': 'Need more data (min 20 points)',
                    'hypothesis': 'H0: No difference between first and second half of data'
                }
        except Exception as e:
            results['anova_test'] = {
                'test': 'One-Way ANOVA Test',
                'statistic': 0,
                'p_value': 1,
                'result': 'Test failed',
                'hypothesis': 'H0: No difference between first and second half of data'
            }
        
        return results
    
    def display_header(self):
        """Display dashboard header"""
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown('<h1 class="main-header">🚀 AI Crowd Analytics Dashboard</h1>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="live-badge">LIVE ANALYTICS</div>', unsafe_allow_html=True)
        with col3:
            st.write(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    
    def display_real_time_metrics(self, data, stats_dict, risk_analysis):
        """Display real-time metrics cards"""
        st.markdown('<div class="section-header">📈 Real-time Monitoring</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Determine alert class based on current count
        current_count = data['current_count']
        thresholds = data.get('alert_thresholds', {'low': 10, 'normal': 20, 'high': 30})
        
        if current_count <= thresholds['low']:
            alert_class = "alert-low"
        elif current_count <= thresholds['normal']:
            alert_class = "alert-normal"
        elif current_count <= thresholds['high']:
            alert_class = "alert-high"
        else:
            alert_class = "alert-critical"
        
        with col1:
            st.markdown(f"""
            <div class="metric-card {alert_class}">
                <div class="metric-content">
                    <div class="metric-title">👥 Current Crowd</div>
                    <div class="metric-value">{data['current_count']}</div>
                    <div class="metric-subtitle">People detected | {data['alert_level']} level</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            alarm_status = "ACTIVE" if data['alarm_active'] else "INACTIVE"
            alarm_class = "alarm-active" if data['alarm_active'] else "alarm-inactive"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-content">
                    <div class="metric-title">🚨 Alert Status</div>
                    <div class="{alarm_class}">{alarm_status}</div>
                    <div class="metric-subtitle">Total alerts: {data['total_alerts']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            risk_score = risk_analysis.get('risk_score', 0)
            risk_color = "#00c853" if risk_score < 30 else "#ffd600" if risk_score < 70 else "#ff1744"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-content">
                    <div class="metric-title">📊 Risk Assessment</div>
                    <div class="metric-value" style="color: {risk_color};">{risk_score}%</div>
                    <div class="metric-subtitle">Anomaly detection score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            frames_processed = data.get('frames_processed', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-content">
                    <div class="metric-title">📈 Processing</div>
                    <div class="metric-value" style="color: #1f77b4;">{frames_processed}</div>
                    <div class="metric-subtitle">Frames analyzed</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def display_alert_status(self, data):
        """Display alert status"""
        st.subheader("🚨 Alert Status")
        
        if data['alarm_active']:
            st.error(f"""
            ## 🔴 CRITICAL ALERT ACTIVE!
            **{data['current_count']}** people detected - **{data['alert_level']}** level
            *Timestamp: {data['timestamp']}*
            
            **⚠️ WARNING:** Crowd density exceeds safety thresholds!
            """)
        else:
            st.success(f"""
            ## 🟢 SYSTEM NORMAL
            **{data['current_count']}** people detected - **{data['alert_level']}** level
            *Last update: {data['timestamp']}*
            
            **✅ SAFE:** Crowd density within acceptable limits
            """)
    
    def display_analytical_charts(self, data, stats_dict, predictions, risk_analysis):
        """Display analytical charts and visualizations"""
        st.markdown('<div class="section-header">📊 Advanced Analytics</div>', unsafe_allow_html=True)
        
        # Row 1: Real-time trend and predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Real-time Crowd Trend")
            if data['people_history'] and len(data['people_history']) > 0:
                # Limit history to last 50 points for better visualization
                history = data['people_history'][-50:]
                thresholds = data.get('alert_thresholds', {'low': 10, 'normal': 20, 'high': 30})
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(history))),
                    y=history,
                    mode='lines+markers',
                    name='People Count',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=4)
                ))
                
                # Add trend line if available
                if predictions.get('trend_line') and len(predictions['trend_line']) >= len(history):
                    fig.add_trace(go.Scatter(
                        x=list(range(len(history))),
                        y=predictions['trend_line'][-len(history):],
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                
                # Add threshold lines
                fig.add_hline(y=thresholds['low'], line_dash="dash", line_color="green", 
                            annotation_text=f"Low ({thresholds['low']})")
                fig.add_hline(y=thresholds['normal'], line_dash="dash", line_color="orange", 
                            annotation_text=f"High ({thresholds['normal']})")
                fig.add_hline(y=thresholds['high'], line_dash="dash", line_color="red", 
                            annotation_text=f"Critical ({thresholds['high']})")
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    xaxis_title="Frame Sequence",
                    yaxis_title="Number of People",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Waiting for crowd data...")
        
        with col2:
            st.subheader("🎯 Alert Distribution")
            if stats_dict.get('alert_distribution'):
                alert_data = stats_dict['alert_distribution']
                
                fig_pie = px.pie(
                    values=list(alert_data.values()),
                    names=list(alert_data.keys()),
                    color=list(alert_data.keys()),
                    color_discrete_map={
                        'LOW': '#00c853',
                        'NORMAL': '#ffd600',
                        'HIGH': '#ff9100',
                        'CRITICAL': '#ff1744'
                    }
                )
                fig_pie.update_layout(height=400, showlegend=True, title="Distribution of Alert Levels")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("📈 Waiting for alert data...")
        
        # Row 2: Statistical analysis
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("📊 Statistical Distribution")
            if data['people_history'] and len(data['people_history']) > 0:
                fig = px.histogram(
                    x=data['people_history'],
                    nbins=20,
                    title="Distribution of Crowd Sizes",
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(height=400, xaxis_title="Number of People", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📋 Waiting for statistical data...")
        
        with col4:
            st.subheader("⚠️ Risk & Anomaly Detection")
            if data['people_history'] and len(data['people_history']) > 0:
                fig = go.Figure()
                
                # Main data
                fig.add_trace(go.Scatter(
                    x=list(range(len(data['people_history']))),
                    y=data['people_history'],
                    mode='lines+markers',
                    name='Crowd Count',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=3)
                ))
                
                # Highlight anomalies
                anomalies = risk_analysis.get('anomalies', [])
                if anomalies:
                    anomaly_values = [data['people_history'][i] for i in anomalies]
                    fig.add_trace(go.Scatter(
                        x=anomalies,
                        y=anomaly_values,
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Frame Sequence",
                    yaxis_title="Number of People",
                    template="plotly_white",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⚠️ Waiting for risk analysis data...")
        
        # NEW ROW: Box plot, Scatter plot, and Correlation Matrix
        st.markdown('<div class="section-header">📊 Advanced Statistical Visualizations</div>', unsafe_allow_html=True)
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.subheader("📦 Box Plot - Crowd Distribution")
            if data['people_history'] and len(data['people_history']) > 0:
                fig_box = px.box(
                    y=data['people_history'],
                    title="Box Plot of Crowd Sizes",
                    color_discrete_sequence=['#1f77b4']
                )
                fig_box.update_layout(
                    height=400,
                    yaxis_title="Number of People",
                    showlegend=False
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("📦 Waiting for crowd data...")
        
        with col6:
            st.subheader("🔍 Scatter Plot - Time vs Crowd")
            if data['people_history'] and len(data['people_history']) > 0:
                # Create scatter plot with trend line
                scatter_data = pd.DataFrame({
                    'Time': range(len(data['people_history'])),
                    'People': data['people_history']
                })
                
                fig_scatter = px.scatter(
                    scatter_data,
                    x='Time',
                    y='People',
                    title="Scatter Plot: Time vs Crowd Size",
                    trendline="lowess",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_scatter.update_layout(
                    height=400,
                    xaxis_title="Time (Frame Sequence)",
                    yaxis_title="Number of People"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("🔍 Waiting for crowd data...")
        
        with col7:
            st.subheader("🔄 Correlation Matrix")
            if data['people_history'] and len(data['people_history']) > 0:
                # Create a simple dataframe for correlation analysis
                if len(data['people_history']) >= 10:
                    # Create lagged variables for correlation analysis - FIXED LENGTH ISSUE
                    df_corr = pd.DataFrame({
                        'Current': data['people_history'][1:],
                        'Lag_1': data['people_history'][:-1]  # One period lag
                    })
                    
                    # Add moving averages for additional correlation insights - FIXED LENGTH ISSUE
                    if len(data['people_history']) >= 5:
                        # Ensure proper length matching for moving average
                        ma_3 = pd.Series(data['people_history']).rolling(window=3).mean().dropna().tolist()
                        # Align lengths by taking appropriate slices
                        min_length = min(len(df_corr), len(ma_3))
                        df_corr = df_corr.iloc[:min_length].copy()
                        df_corr['MA_3'] = ma_3[:min_length]
                    
                    # Calculate correlation matrix
                    corr_matrix = df_corr.corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix"
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("📈 Need more data points for correlation analysis")
            else:
                st.info("🔄 Waiting for crowd data...")
    
    def display_hypothesis_testing(self, data):
        """Display hypothesis testing results"""
        st.markdown('<div class="section-header">🔬 Hypothesis Testing</div>', unsafe_allow_html=True)
        
        # Perform hypothesis tests
        hypothesis_results = self.hypothesis_testing(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistical Tests")
            
            if hypothesis_results:
                for test_name, result in hypothesis_results.items():
                    if result.get('result') != 'Insufficient data' and result.get('result') != 'Test failed':
                        st.markdown(f"**{result['test']}**")
                        st.write(f"**Hypothesis:** {result['hypothesis']}")
                        st.write(f"**Test Statistic:** {result['statistic']:.4f}")
                        st.write(f"**P-value:** {result['p_value']:.4f}")
                        
                        # Display result with color coding
                        if result['p_value'] < 0.05:
                            st.markdown(f'<div class="hypothesis-result hypothesis-rejected">❌ REJECT H0: {result["result"]}</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="hypothesis-result hypothesis-accepted">✅ FAIL TO REJECT H0: {result["result"]}</div>', 
                                      unsafe_allow_html=True)
                        st.divider()
            else:
                st.info("🔬 Perform hypothesis tests using the controls on the right")
        
        with col2:
            st.subheader("🎛️ Testing Controls")
            
            # Interactive hypothesis testing controls
            st.write("**Configure Hypothesis Tests:**")
            
            # Expected mean for T-test
            expected_mean = st.slider("Expected Mean for T-Test", 5, 30, 15, 
                                    help="Set the expected average crowd size for T-test")
            
            # Significance level
            alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01,
                            help="Set the significance level for hypothesis tests")
            
            st.divider()
            
            # Test interpretation guide
            st.subheader("📖 Interpretation Guide")
            st.info("""
            **P-value Interpretation:**
            - **P < α**: Reject null hypothesis (significant result)
            - **P ≥ α**: Fail to reject null hypothesis
            
            **Tests Included:**
            - **T-Test**: Tests if mean differs from expected value
            - **Chi-Square**: Tests if distribution differs from expected pattern
            - **ANOVA**: Tests for differences between data segments
            """)
            
            # Quick stats for reference
            if data['people_history']:
                st.write("**Current Data Summary:**")
                st.write(f"• Sample Size: {len(data['people_history'])}")
                st.write(f"• Current Mean: {np.mean(data['people_history']):.2f}")
                st.write(f"• Current Variance: {np.var(data['people_history']):.2f}")
    
    def display_statistical_insights(self, stats_dict, predictions, risk_analysis):
        """Display statistical insights"""
        st.markdown('<div class="section-header">📋 Statistical Insights</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Descriptive Statistics")
            if stats_dict:
                metrics_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Data Points'],
                    'Value': [
                        f"{stats_dict['mean']:.2f}",
                        f"{stats_dict['median']:.2f}",
                        f"{stats_dict['std_dev']:.2f}",
                        f"{stats_dict['min']}",
                        f"{stats_dict['max']}",
                        f"{stats_dict['total_frames']}"
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.info("No statistics available")
        
        with col2:
            st.subheader("📊 Trend Analysis")
            if predictions:
                trend = predictions.get('trend', 0)
                trend_direction = "📈 Increasing" if trend > 0.1 else "📉 Decreasing" if trend < -0.1 else "➡️ Stable"
                
                trend_df = pd.DataFrame({
                    'Metric': ['Trend Direction', 'Trend Slope', 'Future Predictions'],
                    'Value': [
                        trend_direction,
                        f"{trend:.3f}",
                        f"{len(predictions.get('future_predictions', []))} steps"
                    ]
                })
                st.dataframe(trend_df, use_container_width=True, hide_index=True)
            else:
                st.info("No trend data available")
        
        with col3:
            st.subheader("⚠️ Risk Metrics")
            if risk_analysis:
                risk_df = pd.DataFrame({
                    'Risk Level': ['Low', 'Medium', 'High', 'Anomalies'],
                    'Count': [
                        risk_analysis['low_risk'],
                        risk_analysis['medium_risk'],
                        risk_analysis['high_risk'],
                        len(risk_analysis['anomalies'])
                    ]
                })
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            else:
                st.info("No risk metrics available")
    
    def display_data_export(self, data):
        """Display data export section - FIXED DUPLICATE ISSUE"""
        st.markdown('<div class="section-header">📤 Data Management</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Recent Activity")
            if data['people_history'] and len(data['people_history']) > 0:
                recent_counts = data['people_history'][-10:]
                recent_data = pd.DataFrame({
                    'Frame': [f"#{i+1}" for i in range(len(recent_counts))],
                    'People': recent_counts,
                    'Status': ['LOW' if x <= 10 else 'NORMAL' if x <= 20 else 'HIGH' if x <= 30 else 'CRITICAL' 
                             for x in recent_counts]
                })
                st.dataframe(recent_data, use_container_width=True, height=300)
            else:
                st.info("No activity data available")
        
        with col2:
            st.subheader("⚙️ Controls")
            control_col1, control_col2 = st.columns(2)
            
            with control_col1:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.rerun()
                
                if st.button("🎮 Simulate Data", use_container_width=True):
                    self.simulate_crowd_data()
                    st.rerun()
            
            with control_col2:
                if st.button("📊 Export Data", use_container_width=True):
                    self.export_data(data)
                
                if st.button("🔄 Reset Data", use_container_width=True):
                    self.initialize_data_file()
                    st.rerun()
            
            # MOVED SYSTEM INFORMATION HERE TO AVOID DUPLICATION
            st.subheader("System Information")
            st.write(f"**Video File:** {data.get('video_file', 'Not specified')}")
            st.write(f"**Frames Processed:** {len(data['people_history'])}")
            st.write(f"**Last Update:** {data['timestamp']}")
            
            if data['current_count'] > 0:
                st.success("✅ **System Status:** Active")
            else:
                st.warning("⚠️ **System Status:** Waiting for data")
    
    def export_data(self, data):
        """Export data to CSV"""
        if data['people_history'] and len(data['people_history']) > 0:
            df = pd.DataFrame({
                'frame_number': range(1, len(data['people_history']) + 1),
                'people_count': data['people_history'],
                'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(data['people_history'])
            })
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"crowd_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No data available to export")
    
    def display_sidebar(self):
        """Display sidebar with navigation and information"""
        with st.sidebar:
            st.title("🔧 Analytics Navigation")
            st.divider()
            
            st.subheader("Analysis Types")
            st.info("""
            **Real-time Monitoring:**
            - Live crowd counting
            - Alert status
            - Risk assessment
            
            **Advanced Analytics:**
            - Statistical analysis
            - Predictive modeling
            - Anomaly detection
            - Box plots & Scatter plots
            - Correlation analysis
            - Hypothesis testing (T-test, Chi-square, ANOVA)
            """)
            
            st.divider()
            st.subheader("Alert Thresholds")
            st.write("🟢 **LOW**: 0-10 people")
            st.write("🟡 **NORMAL**: 11-20 people") 
            st.write("🟠 **HIGH**: 21-30 people")
            st.write("🔴 **CRITICAL**: 31+ people")
            
            st.divider()
            st.subheader("Alert Status Logic")
            st.write("✅ **ACTIVE**: When people count > 20")
            st.write("✅ **INACTIVE**: When people count ≤ 20")
            
            st.divider()
            st.subheader("How to Use")
            st.write("1. Run crowd detection system")
            st.write("2. Monitor real-time analytics")
            st.write("3. Check statistical insights")
            st.write("4. Export data for reporting")
    
    def main(self):
        """Main dashboard function"""
        # Load current real-time data
        data = self.load_data()
        
        # Perform analytics
        stats_dict = self.calculate_statistics(data)
        predictions = self.predictive_analysis(data)
        risk_analysis = self.risk_analysis(data)
        
        # Display all components
        self.display_header()
        self.display_real_time_metrics(data, stats_dict, risk_analysis)
        self.display_alert_status(data)
        st.divider()
        self.display_analytical_charts(data, stats_dict, predictions, risk_analysis)
        self.display_hypothesis_testing(data)
        self.display_statistical_insights(stats_dict, predictions, risk_analysis)
        self.display_data_export(data)  # Only called once now
        self.display_sidebar()

# Create and run dashboard
if __name__ == "__main__":
    dashboard = IntegratedCrowdAnalyticsDashboard()
    
    # Auto-refresh every 3 seconds
    while True:
        dashboard.main()
        time.sleep(3)
        st.rerun()