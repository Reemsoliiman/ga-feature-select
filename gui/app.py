"""
Enhanced GUI Application for GA Feature Selection
Features: 3D Plotly visualizations, dataset validation, full experiments mode
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from itertools import product

from src.genetic_algorithm import GeneticAlgorithm
from src.utils import load_dataset, split_data
from src.decision_tree import DecisionTreeWrapper
from src.config import (
    POPULATION_SIZE, N_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE,
    SELECTION_METHODS, CROSSOVER_METHODS, MUTATION_METHODS, WEIGHT_THRESHOLD,
    N_RUNS, RANDOM_SEED
)

# Page configuration
st.set_page_config(
    page_title="GA Feature Selection",
    page_icon="DNA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    /* UPDATED CARD STYLE */
    .info-card {
        background-color: white;
        padding: 1.5rem;       /* Reduced padding to fit more text */
        border-radius: 12px;
        border: 1px solid #e9ecef;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        height: 380px;         /* Increased height to prevent overflow */
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .info-card h4 {
        color: #667eea;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    .info-card ul {
        list-style-type: none;
        padding-left: 0;
        margin: 0;
        flex-grow: 1;
    }
    .info-card ul li {
        margin-bottom: 0.5rem; /* Reduced spacing between items */
        font-size: 0.95rem;
        color: #4a5568;
        padding-left: 1.2rem;
        position: relative;
    }
    .info-card ul li::before {
        content: "•";
        color: #667eea;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def validate_dataset(df):
    """Validate dataset structure and quality"""
    issues = []
    warnings = []
    
    # Check if empty
    if df.empty:
        issues.append("Dataset is empty")
        return False, issues, warnings
    
    # Check minimum rows
    if len(df) < 50:
        warnings.append(f"Small dataset: only {len(df)} samples (recommended: 50+)")
    
    # Check minimum columns
    if df.shape[1] < 2:
        issues.append("Dataset must have at least 2 columns (features + target)")
        return False, issues, warnings
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"Columns with all null values: {null_cols}")
    
    # Check last column (target)
    target_col = df.columns[-1]
    target_nulls = df[target_col].isnull().sum()
    if target_nulls > 0:
        issues.append(f"Target column '{target_col}' has {target_nulls} null values")
    
    # Check target cardinality
    n_unique = df[target_col].nunique()
    if n_unique < 2:
        issues.append(f"Target column has only {n_unique} unique value(s) - need at least 2 classes")
    elif n_unique > 50:
        warnings.append(f"Target has {n_unique} unique values - regression not supported, will be treated as classification")
    
    # Check for constant features
    constant_features = [col for col in df.columns[:-1] if df[col].nunique() == 1]
    if constant_features:
        warnings.append(f"Constant features (no variance): {len(constant_features)} columns")
    
    # Check data types
    non_numeric = df.iloc[:, :-1].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        warnings.append(f"Non-numeric features: {len(non_numeric)} columns (will be encoded)")
    
    is_valid = len(issues) == 0
    return is_valid, issues, warnings

def create_3d_convergence_plot(history):
    """Create 3D surface plot of fitness evolution"""
    generations = np.arange(len(history['best_fitness']))
    
    fig = go.Figure()
    
    # Best fitness line
    fig.add_trace(go.Scatter3d(
        x=generations,
        y=history['best_fitness'],
        z=history['n_selected_features'],
        mode='lines+markers',
        name='Best Fitness',
        line=dict(color='#667eea', width=4),
        marker=dict(size=4, color='#667eea')
    ))
    
    # Mean fitness line
    fig.add_trace(go.Scatter3d(
        x=generations,
        y=history['mean_fitness'],
        z=history['n_selected_features'],
        mode='lines+markers',
        name='Mean Fitness',
        line=dict(color='#764ba2', width=3, dash='dash'),
        marker=dict(size=3, color='#764ba2')
    ))
    
    fig.update_layout(
        title='3D Evolution Trajectory: Fitness vs Generation vs Features',
        scene=dict(
            xaxis_title='Generation',
            yaxis_title='Fitness',
            zaxis_title='Features Selected',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    # Enable zoom, pan, rotate
    fig.update_layout(
        dragmode='orbit',
        scene=dict(
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False)
        )
    )
    
    return fig

def create_feature_importance_3d(feature_names, weights, selected_mask):
    """Create 3D scatter plot of feature importance"""
    indices = np.arange(len(feature_names))
    colors = ['#667eea' if s else '#ddd' for s in selected_mask]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=indices,
        y=weights,
        z=selected_mask.astype(int),
        mode='markers+text',
        marker=dict(
            size=weights * 15 + 5,
            color=colors,
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        text=[f"{name}<br>w={w:.3f}" for name, w in zip(feature_names, weights)],
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>Index: %{x}<br>Weight: %{y:.4f}<extra></extra>'
    )])
    
    fig.update_layout(
        title='3D Feature Space: Index vs Weight vs Selection',
        scene=dict(
            xaxis_title='Feature Index',
            yaxis_title='Weight',
            zaxis_title='Selected (0/1)',
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.3))
        ),
        height=700,
        showlegend=False,
        hovermode='closest'
    )
    
    # Enable zoom, pan, rotate
    fig.update_layout(
        dragmode='orbit',
        scene=dict(
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False)
        )
    )
    
    return fig

def create_interactive_convergence(history):
    """Create interactive dual-axis convergence plot"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    generations = list(range(len(history['best_fitness'])))
    
    # Fitness traces
    fig.add_trace(
        go.Scatter(x=generations, y=history['best_fitness'], 
                   name='Best Fitness', line=dict(color='#667eea', width=3)),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=generations, y=history['mean_fitness'], 
                   name='Mean Fitness', line=dict(color='#764ba2', width=2, dash='dash')),
        secondary_y=False
    )
    
    # Features trace
    fig.add_trace(
        go.Scatter(x=generations, y=history['n_selected_features'], 
                   name='Features Selected', line=dict(color='#f093fb', width=2)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Fitness", secondary_y=False)
    fig.update_yaxes(title_text="Features Selected", secondary_y=True)
    
    fig.update_layout(
        title='Evolution Progress: Fitness & Feature Reduction',
        hovermode='x unified',
        height=500,
        dragmode='zoom'
    )
    
    # Configure zoom and pan
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    
    return fig

def run_full_experiments(X_train, y_train, X_test, y_test, feature_names):
    """Run all 27 operator combinations"""
    configs = list(product(SELECTION_METHODS, CROSSOVER_METHODS, MUTATION_METHODS))
    total_runs = len(configs) * N_RUNS
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (selection, crossover, mutation) in enumerate(configs):
        config_name = f"{selection}_{crossover}_{mutation}"
        
        for run in range(N_RUNS):
            status_text.text(f"Running {config_name} - Run {run+1}/{N_RUNS}")
            
            # Run GA
            ga = GeneticAlgorithm(
                X_train, y_train,
                selection=selection,
                crossover=crossover,
                mutation=mutation,
                random_state=RANDOM_SEED + run
            )
            best_weights, history = ga.evolve()
            
            # Evaluate on test set
            best_mask = (best_weights >= WEIGHT_THRESHOLD).astype(int)
            selected_indices = np.where(best_mask == 1)[0]
            
            if len(selected_indices) > 0:
                X_test_selected = X_test[:, selected_indices]
                X_train_selected = X_train[:, selected_indices]
                dt = DecisionTreeWrapper()
                dt.train(X_train_selected, y_train)
                test_accuracy = dt.evaluate(X_test_selected, y_test)
            else:
                test_accuracy = 0.0
            
            results.append({
                'selection': selection,
                'crossover': crossover,
                'mutation': mutation,
                'run': run + 1,
                'train_fitness': history['best_fitness'][-1],
                'test_accuracy': test_accuracy,
                'n_features': int(np.sum(best_mask)),
                'reduction_pct': (1 - np.sum(best_mask)/len(best_weights)) * 100
            })
            
            progress = ((idx * N_RUNS) + run + 1) / total_runs
            progress_bar.progress(progress)
    
    status_text.text("All experiments complete!")
    return pd.DataFrame(results)

# Main App
st.markdown('<h1 class="main-header">Genetic Algorithm Feature Selection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Optimize your machine learning features with evolutionary algorithms</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            # 1. LOAD DATA INTO SESSION STATE (To allow editing)
            # We check if the file changed or if 'df' is missing to reload
            if 'df' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
                st.session_state['df'] = pd.read_csv(uploaded_file)
                st.session_state['current_file'] = uploaded_file.name
            
            # Use the dataframe from session state
            df = st.session_state['df']
            
            # --- INTERACTIVE CLEANING & PREP ---
            st.subheader("1. Dataset Preparation")
            
            # A. SELECT TARGET COLUMN
            columns = df.columns.tolist()
            target_col = st.selectbox(
                "Select Target Column", 
                columns, 
                index=len(columns)-1,
                help="The column you want to predict."
            )
            
            # B. CHECK: TARGET POSITION
            # The app expects target to be the last column. Check if it is.
            if df.columns[-1] != target_col:
                st.warning(f"Target '{target_col}' is not the last column.")
                if st.button("Fix: Move Target to End", type="primary"):
                    # Reorder columns: everything else + target
                    new_order = [c for c in df.columns if c != target_col] + [target_col]
                    st.session_state['df'] = df[new_order]
                    st.rerun()

            # C. CHECK: 100% EMPTY COLUMNS
            # Find columns that are all null
            null_cols = df.columns[df.isnull().all()].tolist()
            if null_cols:
                st.error(f"Found {len(null_cols)} completely empty columns.")
                with st.expander("View Empty Columns"):
                    st.write(null_cols)
                
                if st.button("Fix: Drop Empty Columns", type="primary"):
                    st.session_state['df'] = df.drop(columns=null_cols)
                    st.rerun()

            # D. CHECK: MISSING TARGET VALUES
            # Count rows where target is NaN
            missing_target_count = df[target_col].isnull().sum()
            if missing_target_count > 0:
                st.error(f"Target column has {missing_target_count} missing values.")
                if st.button("Fix: Drop Rows with Missing Target", type="primary"):
                    st.session_state['df'] = df.dropna(subset=[target_col])
                    st.rerun()

            # --- VALIDATION ---
            # Now we validate the CLEANED dataframe
            df = st.session_state['df'] # Refresh ref
            
            is_valid, issues, warnings = validate_dataset(df)
            
            if not is_valid:
                st.error("Dataset Validation Failed")
                for issue in issues:
                    st.error(f"ERROR: {issue}")
                st.stop()
            
            if warnings:
                with st.expander("Validation Warnings", expanded=True):
                    for warning in warnings:
                        st.warning(warning)
            
            st.success(f"Dataset Ready: {df.shape[0]} rows, {df.shape[1]} cols")
            
            # --- MODE SELECTION ---
            st.markdown("---")
            st.subheader("2. Run Configuration")
            mode = st.radio(
                "Execution Mode",
                ["Single Run (Custom Parameters)", "Full Experiments (27 Configs)"],
                help="Single Run: Configure and test one operator combination\nFull Experiments: Run all 27 operator combinations"
            )
            
            if mode == "Single Run (Custom Parameters)":
                st.markdown("##### GA Parameters")
                pop_size = st.slider("Population Size", 20, 200, POPULATION_SIZE, 10)
                n_gen = st.slider("Generations", 20, 500, N_GENERATIONS, 10)
                # Feature Penalty Slider (Added based on previous discussion)
                lambda_val = st.slider("Feature Penalty (Lambda)", 0.0, 0.5, 0.05, 0.01, 
                       help="Higher = Fewer features selected")
                mut_rate = st.slider("Mutation Rate", 0.001, 0.1, MUTATION_RATE, 0.001, format="%.3f")
                cross_rate = st.slider("Crossover Rate", 0.5, 1.0, CROSSOVER_RATE, 0.05)
                
                st.markdown("##### Operators")
                selection_method = st.selectbox("Selection", SELECTION_METHODS)
                crossover_method = st.selectbox("Crossover", CROSSOVER_METHODS)
                mutation_method = st.selectbox("Mutation", MUTATION_METHODS)
                
                run_button = st.button("Run Single Experiment", type="primary")
                run_full = False
            else:
                st.info(f"Experiments: {len(SELECTION_METHODS) * len(CROSSOVER_METHODS) * len(MUTATION_METHODS)} configs × {N_RUNS} runs")
                run_full = st.button("Start Full Experiments", type="primary")
                run_button = False
            
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

# Main content
if uploaded_file:
    with st.expander("Dataset Preview & Statistics", expanded=False):
        # UPDATED: Changed ratio to 2:1 (Dataframe takes 2/3, Stats takes 1/3)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("##### Raw Data")
            st.dataframe(df.head(10), use_container_width=True)
            
        with col2:
            st.markdown("##### Metadata")
            # Using a container with border for the stats to make it look cleaner
            with st.container(border=True):
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write(f"**Target:** `{df.columns[-1]}`")
                st.write(f"**Classes:** {df.iloc[:, -1].nunique()}")
                
                missing = df.isnull().sum().sum()
                if missing > 0:
                    st.error(f"**Missing Values:** {missing}")
                else:
                    st.success("**Missing Values:** None")
    
    if run_button:
        # Prepare data
        temp_path = "temp_dataset.csv"
        df.to_csv(temp_path, index=False)
        X, y, feature_names = load_dataset(temp_path)
        os.remove(temp_path)
        
        # Convert to numpy arrays immediately
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_SEED)
        
        with st.spinner(f"Running GA for {n_gen} generations..."):
            ga = GeneticAlgorithm(
                X_train, y_train,
                selection=selection_method,
                crossover=crossover_method,
                mutation=mutation_method,
                pop_size=pop_size,
                n_generations=n_gen,
                mut_rate=mut_rate,
                cross_rate=cross_rate
            )
            best_weights, history = ga.evolve()
        
        # Evaluate on test set
        best_mask = (best_weights >= WEIGHT_THRESHOLD).astype(int)
        n_selected = int(np.sum(best_mask))
        selected_indices = np.where(best_mask == 1)[0]
        
        # Convert to numpy arrays
        X_train_np = np.array(X_train)
        X_test_np = np.array(X_test)
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)
        
        if len(selected_indices) > 0:
            X_test_selected = X_test_np[:, selected_indices]
            X_train_selected = X_train_np[:, selected_indices]
            dt = DecisionTreeWrapper()
            dt.train(X_train_selected, y_train_np)
            test_accuracy = dt.evaluate(X_test_selected, y_test_np)
        else:
            test_accuracy = 0.0
        
        # Display Results
        st.markdown("---")
        st.header("Results")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train Fitness", f"{history['best_fitness'][-1]:.4f}")
        col2.metric("Test Accuracy", f"{test_accuracy:.4f}")
        col3.metric("Features Selected", f"{n_selected}/{len(feature_names)}")
        col4.metric("Reduction", f"{(1 - n_selected/len(feature_names)) * 100:.1f}%")
        
        # 3D Visualizations
        st.subheader("Interactive 3D Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["3D Evolution", "Feature Importance 3D", "Convergence Plot"])
        
        with tab1:
            fig_3d = create_3d_convergence_plot(history)
            st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['pan3d', 'zoom3d', 'orbitRotation', 'resetCameraDefault3d']})
        
        with tab2:
            fig_features = create_feature_importance_3d(feature_names, best_weights, best_mask)
            st.plotly_chart(fig_features, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['pan3d', 'zoom3d', 'orbitRotation', 'resetCameraDefault3d']})
        
        with tab3:
            fig_conv = create_interactive_convergence(history)
            st.plotly_chart(fig_conv, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'resetScale2d', 'autoScale2d']})
        
        # Selected Features (ranked by weight)
        st.subheader("Selected Features (Ranked by Weight)")
        
        # Create dataframe with all features and their weights
        feature_data = []
        for i, (name, weight) in enumerate(zip(feature_names, best_weights)):
            feature_data.append({
                'Rank': None,
                'Feature': name,
                'Weight': weight,
                'Selected': 'Yes' if best_mask[i] else 'No',
                'Index': i
            })
        
        # Sort by weight (descending)
        feature_df = pd.DataFrame(feature_data).sort_values('Weight', ascending=False)
        
        # Add rank only for selected features
        selected_df = feature_df[feature_df['Selected'] == 'Yes'].copy()
        selected_df['Rank'] = range(1, len(selected_df) + 1)
        
        # Display selected features
        st.dataframe(
            selected_df[['Rank', 'Feature', 'Weight', 'Index']].style.background_gradient(
                subset=['Weight'], cmap='viridis'
            ),
            use_container_width=True
        )
        
        # Download results
        st.subheader("Export Results")
        
        results_dict = {
            'configuration': {
                'selection': selection_method,
                'crossover': crossover_method,
                'mutation': mutation_method,
                'population_size': pop_size,
                'generations': n_gen,
                'mutation_rate': mut_rate,
                'crossover_rate': cross_rate
            },
            'results': {
                'train_fitness': float(history['best_fitness'][-1]),
                'test_accuracy': float(test_accuracy),
                'features_selected': int(n_selected),
                'total_features': len(feature_names),
                'reduction_percentage': float((1 - n_selected/len(feature_names)) * 100)
            },
            'selected_features': selected_df[['Rank', 'Feature', 'Weight']].to_dict('records')
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Results (JSON)",
                json.dumps(results_dict, indent=2),
                f"ga_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
        with col2:
            st.download_button(
                "Download Features (CSV)",
                selected_df[['Rank', 'Feature', 'Weight', 'Index']].to_csv(index=False),
                f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    elif run_full:
        # Prepare data
        temp_path = "temp_dataset.csv"
        df.to_csv(temp_path, index=False)
        X, y, feature_names = load_dataset(temp_path)
        os.remove(temp_path)
        
        # Convert to numpy arrays immediately
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_SEED)
        
        st.info("Running full experiments - this may take several minutes...")
        
        results_df = run_full_experiments(X_train, y_train, X_test, y_test, feature_names)
        
        st.success("All experiments complete!")
        
        # Summary statistics
        st.header("Experiment Results Summary")
        
        summary = results_df.groupby(['selection', 'crossover', 'mutation']).agg({
            'test_accuracy': ['mean', 'std'],
            'n_features': 'mean',
            'reduction_pct': 'mean'
        }).round(4)
        
        st.dataframe(summary, use_container_width=True)
        
        # Best configuration
        best_config = results_df.loc[results_df['test_accuracy'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Config", f"{best_config['selection']}+{best_config['crossover']}+{best_config['mutation']}")
        col2.metric("Test Accuracy", f"{best_config['test_accuracy']:.4f}")
        col3.metric("Features Used", f"{int(best_config['n_features'])}/{len(feature_names)}")
        
        # === COMPREHENSIVE OPERATOR COMPARISON VISUALIZATIONS ===
        st.markdown("---")
        st.header("Detailed Operator Comparison Analysis")
        
        # Tab organization for different visualizations
        viz_tabs = st.tabs([
            "Box Plots", 
            "Heatmaps", 
            "Feature Reduction", 
            "Performance Distribution",
            "Statistical Summary"
        ])
        
        # TAB 1: Box Plots (Original + Enhanced)
        with viz_tabs[0]:
            st.subheader("Performance Distribution by Operators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.box(results_df, x='selection', y='test_accuracy', color='crossover',
                             title='Test Accuracy by Selection Method',
                             labels={'test_accuracy': 'Test Accuracy', 'selection': 'Selection Method'},
                             height=450)
                fig1.update_layout(dragmode='zoom')
                fig1.update_xaxes(fixedrange=False)
                fig1.update_yaxes(fixedrange=False)
                st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            
            with col2:
                fig2 = px.box(results_df, x='mutation', y='test_accuracy', color='selection',
                             title='Test Accuracy by Mutation Method',
                             labels={'test_accuracy': 'Test Accuracy', 'mutation': 'Mutation Method'},
                             height=450)
                fig2.update_layout(dragmode='zoom')
                fig2.update_xaxes(fixedrange=False)
                fig2.update_yaxes(fixedrange=False)
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        
        # TAB 2: Heatmaps (One for each selection method)
        with viz_tabs[1]:
            st.subheader("Performance Heatmaps: Crossover vs Mutation")
            st.markdown("*Average test accuracy across all runs for each operator combination*")
            
            heatmap_cols = st.columns(len(SELECTION_METHODS))
            
            for idx, sel_method in enumerate(SELECTION_METHODS):
                with heatmap_cols[idx]:
                    # Filter data for this selection method
                    filtered = results_df[results_df['selection'] == sel_method]
                    
                    # Create pivot table
                    pivot = filtered.pivot_table(
                        values='test_accuracy',
                        index='mutation',
                        columns='crossover',
                        aggfunc='mean'
                    )
                    
                    # Create heatmap
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns,
                        y=pivot.index,
                        colorscale='Viridis',
                        text=np.round(pivot.values, 4),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="Accuracy")
                    ))
                    
                    fig_heat.update_layout(
                        title=f'{sel_method.capitalize()} Selection',
                        xaxis_title='Crossover',
                        yaxis_title='Mutation',
                        height=400,
                        margin=dict(l=80, r=20, t=60, b=60)
                    )
                    
                    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
        
        # TAB 3: Feature Reduction Analysis
        with viz_tabs[2]:
            st.subheader("Feature Reduction Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter: Accuracy vs Features
                fig_scatter = px.scatter(
                    results_df,
                    x='n_features',
                    y='test_accuracy',
                    color='selection',
                    symbol='crossover',
                    size='reduction_pct',
                    title='Test Accuracy vs Number of Features Selected',
                    labels={'n_features': 'Features Selected', 'test_accuracy': 'Test Accuracy'},
                    height=500,
                    hover_data=['mutation', 'reduction_pct']
                )
                fig_scatter.update_layout(dragmode='zoom')
                st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            
            with col2:
                # Bar: Average reduction by operator
                avg_reduction = results_df.groupby('selection')['reduction_pct'].mean().reset_index()
                fig_bar = px.bar(
                    avg_reduction,
                    x='selection',
                    y='reduction_pct',
                    title='Average Feature Reduction by Selection Method',
                    labels={'reduction_pct': 'Reduction (%)', 'selection': 'Selection Method'},
                    height=500,
                    color='selection',
                    text='reduction_pct'
                )
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
        # TAB 4: Performance Distribution
        with viz_tabs[3]:
            st.subheader("Detailed Performance Distributions")
            
            # Violin plots for each operator type
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_v1 = px.violin(
                    results_df,
                    y='test_accuracy',
                    x='selection',
                    color='selection',
                    box=True,
                    title='Selection Methods',
                    labels={'test_accuracy': 'Test Accuracy'},
                    height=450
                )
                st.plotly_chart(fig_v1, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                fig_v2 = px.violin(
                    results_df,
                    y='test_accuracy',
                    x='crossover',
                    color='crossover',
                    box=True,
                    title='Crossover Methods',
                    labels={'test_accuracy': 'Test Accuracy'},
                    height=450
                )
                st.plotly_chart(fig_v2, use_container_width=True, config={'displayModeBar': False})
            
            with col3:
                fig_v3 = px.violin(
                    results_df,
                    y='test_accuracy',
                    x='mutation',
                    color='mutation',
                    box=True,
                    title='Mutation Methods',
                    labels={'test_accuracy': 'Test Accuracy'},
                    height=450
                )
                st.plotly_chart(fig_v3, use_container_width=True, config={'displayModeBar': False})
        
        # TAB 5: Statistical Summary
        with viz_tabs[4]:
            st.subheader("Statistical Analysis")
            
            # Detailed statistics by operator
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Selection Methods")
                sel_stats = results_df.groupby('selection').agg({
                    'test_accuracy': ['mean', 'std', 'min', 'max'],
                    'n_features': 'mean',
                    'reduction_pct': 'mean'
                }).round(4)
                st.dataframe(sel_stats, use_container_width=True)
            
            with col2:
                st.markdown("##### Crossover Methods")
                cross_stats = results_df.groupby('crossover').agg({
                    'test_accuracy': ['mean', 'std', 'min', 'max'],
                    'n_features': 'mean',
                    'reduction_pct': 'mean'
                }).round(4)
                st.dataframe(cross_stats, use_container_width=True)
            
            with col3:
                st.markdown("##### Mutation Methods")
                mut_stats = results_df.groupby('mutation').agg({
                    'test_accuracy': ['mean', 'std', 'min', 'max'],
                    'n_features': 'mean',
                    'reduction_pct': 'mean'
                }).round(4)
                st.dataframe(mut_stats, use_container_width=True)
            
            # Top 10 configurations
            st.markdown("---")
            st.markdown("##### Top 10 Configurations (by Test Accuracy)")
            top10 = results_df.nlargest(10, 'test_accuracy')[
                ['selection', 'crossover', 'mutation', 'test_accuracy', 'n_features', 'reduction_pct']
            ].reset_index(drop=True)
            top10.index = range(1, 11)
            st.dataframe(top10.style.background_gradient(subset=['test_accuracy'], cmap='Greens'), use_container_width=True)
        
        # Download
        st.download_button(
            "Download Full Results (CSV)",
            results_df.to_csv(index=False),
            f"full_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

else:
    # Landing page with improved layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>Get Started</h2>
            <p style='font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem;'>
                Upload your CSV dataset using the sidebar to begin feature selection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Dataset Requirements</h4>
            <ul>
                <li>CSV format required</li>
                <li>Last column must be target variable</li>
                <li>Minimum 50 samples recommended</li>
                <li>At least 2 classes in target</li>
                <li>Handles missing values automatically</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Two Execution Modes</h4>
            <ul>
                <li><strong>Single Run</strong>: Test custom parameters</li>
                <li><strong>Full Experiments</strong>: Test all 27 configurations</li>
                <li>Interactive 3D visualizations</li>
                <li>Real-time progress tracking</li>
                <li>Export results as JSON/CSV</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>Algorithm Features</h4>
            <ul>
                <li>3 selection methods (tournament, roulette, rank)</li>
                <li>3 crossover operators (single-point, uniform, arithmetic)</li>
                <li>3 mutation strategies (bit-flip, uniform, adaptive)</li>
                <li>Continuous weight encoding</li>
                <li>Train/test split validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Example usage section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; color: white; margin-top: 2rem;'>
        <h3 style='margin-bottom: 1rem;'>How It Works</h3>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;'>
            <div style='text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>1</div>
                <p style='margin: 0;'>Upload CSV dataset</p>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>2</div>
                <p style='margin: 0;'>Choose mode & configure</p>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>3</div>
                <p style='margin: 0;'>Run GA optimization</p>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>4</div>
                <p style='margin: 0;'>Explore 3D results</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d;'>
    <p>GA Feature Selection | Powered by Genetic Algorithms & Plotly</p>
</div>
""", unsafe_allow_html=True)