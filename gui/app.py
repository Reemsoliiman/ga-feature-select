"""
GUI Application for GA Feature Selection
Provides user interface for dataset loading, parameter configuration, and visualization.
Uses Streamlit for simplicity.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from src.genetic_algorithm import GeneticAlgorithm
from src.utils import load_dataset
from src.config import (
    POPULATION_SIZE, N_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE,
    SELECTION_METHODS, CROSSOVER_METHODS, MUTATION_METHODS, WEIGHT_THRESHOLD
)

# Page configuration
st.set_page_config(
    page_title="GA Feature Selection",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 Genetic Algorithm Feature Selection")
st.markdown("---")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=['csv'])

if uploaded_file:
    # Load dataset
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display dataset preview
        with st.expander("📊 Dataset Preview"):
            st.dataframe(df.head(10))
            st.write(f"**Shape**: {df.shape}")
            st.write(f"**Columns**: {', '.join(df.columns.tolist())}")
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")
        st.stop()
    
    # GA Parameters
    st.sidebar.subheader("GA Parameters")
    
    pop_size = st.sidebar.slider("Population Size", 20, 200, POPULATION_SIZE, 10)
    n_gen = st.sidebar.slider("Generations", 20, 500, N_GENERATIONS, 10)
    mut_rate = st.sidebar.slider("Mutation Rate", 0.001, 0.1, MUTATION_RATE, 0.001, format="%.3f")
    cross_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, CROSSOVER_RATE, 0.05)
    
    # Operator selection
    st.sidebar.subheader("Operators")
    selection_method = st.sidebar.selectbox("Selection", SELECTION_METHODS)
    crossover_method = st.sidebar.selectbox("Crossover", CROSSOVER_METHODS)
    mutation_method = st.sidebar.selectbox("Mutation", MUTATION_METHODS)
    
    # Run button
    run_button = st.sidebar.button("🚀 Run GA", type="primary")
    
    if run_button:
        # Prepare data
        with st.spinner("Loading and preprocessing data..."):
            # Save uploaded file temporarily
            temp_path = "temp_dataset.csv"
            df.to_csv(temp_path, index=False)
            
            X, y, feature_names = load_dataset(temp_path)
            
            os.remove(temp_path)
        
        st.success(f"✓ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Run GA
        with st.spinner(f"Running GA for {n_gen} generations..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize GA
            ga = GeneticAlgorithm(
                X, y,
                selection=selection_method,
                crossover=crossover_method,
                mutation=mutation_method,
                pop_size=pop_size,
                n_generations=n_gen,
                mut_rate=mut_rate,
                cross_rate=cross_rate
            )
            
            # Evolution
            best_weights, history = ga.evolve()
            
            progress_bar.progress(100)
            status_text.text("✓ Evolution complete!")
        
        # Convert weights to binary mask
        best_mask = (best_weights >= WEIGHT_THRESHOLD).astype(int)
        n_selected = int(np.sum(best_mask))
        
        # Display Results
        st.markdown("---")
        st.header("📈 Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Fitness", f"{history['best_fitness'][-1]:.6f}")
        
        with col2:
            st.metric("Features Selected", f"{n_selected}/{len(feature_names)}")
        
        with col3:
            reduction = (1 - n_selected/len(feature_names)) * 100
            st.metric("Reduction", f"{reduction:.1f}%")
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Convergence Plot")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(history['best_fitness'], label='Best Fitness', linewidth=2, color='tab:blue')
            ax1.plot(history['mean_fitness'], label='Mean Fitness', linewidth=2, alpha=0.7, color='tab:orange')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Feature Reduction")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(history['n_selected_features'], linewidth=2, color='green')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Features Selected')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
        
        # Selected Features
        st.subheader("Selected Features")
        selected_indices = np.where(best_mask == 1)[0]
        selected_names = [feature_names[i] for i in selected_indices]
        
        st.write(f"**{len(selected_names)} features selected:**")
        
        # Display as dataframe with weights
        feature_df = pd.DataFrame({
            'Feature': selected_names,
            'Index': selected_indices,
            'Weight': [f"{best_weights[i]:.3f}" for i in selected_indices]
        })
        st.dataframe(feature_df, use_container_width=True)
        
        # Download results
        st.subheader("💾 Export Results")
        
        results_dict = {
            'configuration': {
                'selection': selection_method,
                'crossover': crossover_method,
                'mutation': mutation_method,
                'population_size': pop_size,
                'generations': n_gen,
                'mutation_rate': mut_rate,
                'crossover_rate': cross_rate,
                'weight_threshold': WEIGHT_THRESHOLD
            },
            'results': {
                'final_fitness': float(history['best_fitness'][-1]),
                'features_selected': int(n_selected),
                'total_features': int(len(feature_names)),
                'reduction_percentage': float(reduction),
                'selected_features': selected_names,
                'best_weights': best_weights.tolist(),
                'best_mask': best_mask.tolist()
            },
            'history': {
                'best_fitness': [float(f) for f in history['best_fitness']],
                'mean_fitness': [float(f) for f in history['mean_fitness']],
                'n_selected_features': [int(n) for n in history['n_selected_features']]
            }
        }
        
        results_json = json.dumps(results_dict, indent=2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Results (JSON)",
                data=results_json,
                file_name=f"ga_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Create CSV for selected features
            csv = feature_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Features (CSV)",
                data=csv,
                file_name=f"selected_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Upload a CSV dataset to begin")
    
    # Instructions
    st.markdown("""
    ### How to Use
    1. **Upload Dataset**: Click "Browse files" in the sidebar and select a CSV file
    2. **Configure Parameters**: Adjust GA parameters and select operators
    3. **Run**: Click the "Run GA" button
    4. **View Results**: See convergence plots and selected features
    5. **Export**: Download results as JSON or CSV
    
    ### Dataset Requirements
    - CSV format
    - Last column should be the target variable
    - Numeric or categorical features (will be encoded automatically)
    - Missing values will be handled automatically
    
    ### Operator Descriptions
    **Selection Methods**:
    - **Tournament**: Best from random k-sized groups
    - **Roulette**: Probability proportional to fitness
    - **Rank**: Probability based on ranking
    
    **Crossover Methods**:
    - **Single Point**: Cut and swap at random point
    - **Uniform**: Random inheritance per gene
    - **Arithmetic**: Weighted average of parents
    
    **Mutation Methods**:
    - **Bit Flip**: Flip weights (w' = 1 - w)
    - **Uniform**: Replace with random values
    - **Adaptive**: Decreasing mutation rate over time
    
    ### About This Project
    This GA uses **continuous weight encoding** where each feature has a weight in [0, 1].
    Features with weight ≥ 0.5 are considered selected for fitness evaluation.
    
    Fitness = Accuracy - λ × (Features Selected / Total Features)
    
    The algorithm evolves a population of weight vectors through selection, crossover,
    mutation, and elitism to find optimal feature subsets.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧬 GA Feature Selection | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)