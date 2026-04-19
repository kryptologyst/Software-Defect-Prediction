"""Streamlit demo for software defect prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.synthetic_data import generate_synthetic_dataset, create_train_test_split
from models.defect_predictor import DefectPredictor
from evaluation.metrics import DefectPredictionEvaluator
from explainability.shap_explainer import DefectPredictionExplainer
from features.feature_engineering import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🐛 Software Defect Prediction</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Disclaimer</h4>
    <p>This is a <strong>research demonstration</strong> for educational purposes only. 
    Results may be inaccurate and should not be used for production security decisions. 
    This is not a SOC tool.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Data parameters
        st.subheader("Dataset Parameters")
        n_samples = st.slider("Number of samples", 100, 2000, 1000)
        defect_ratio = st.slider("Defect ratio", 0.1, 0.5, 0.3)
        random_state = st.number_input("Random seed", 1, 1000, 42)
        
        # Model parameters
        st.subheader("Model Parameters")
        model_type = st.selectbox(
            "Model type",
            ["random_forest", "xgboost", "lightgbm", "neural_network", "ensemble"]
        )
        
        # Feature engineering
        st.subheader("Feature Engineering")
        use_feature_engineering = st.checkbox("Use feature engineering", True)
        scaler_type = st.selectbox("Scaler type", ["standard", "robust"])
        
        # Evaluation parameters
        st.subheader("Evaluation")
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        
        # Generate button
        generate_button = st.button("🚀 Generate & Train Model", type="primary")
    
    # Main content
    if generate_button:
        with st.spinner("Generating dataset and training model..."):
            
            # Generate dataset
            data = generate_synthetic_dataset(
                n_samples=n_samples,
                defect_ratio=defect_ratio,
                random_state=random_state
            )
            
            # Create train-test split
            split_data = create_train_test_split(data, random_state=random_state)
            
            # Train model
            predictor = DefectPredictor(model_type=model_type, random_state=random_state)
            predictor.fit(
                split_data['X_train'], 
                split_data['y_train'],
                use_feature_engineering=use_feature_engineering
            )
            
            # Make predictions
            train_pred = predictor.predict(split_data['X_train'])
            train_proba = predictor.predict_proba(split_data['X_train'])
            test_pred = predictor.predict(split_data['X_test'])
            test_proba = predictor.predict_proba(split_data['X_test'])
            
            # Store in session state
            st.session_state.data = data
            st.session_state.split_data = split_data
            st.session_state.predictor = predictor
            st.session_state.train_pred = train_pred
            st.session_state.train_proba = train_proba
            st.session_state.test_pred = test_pred
            st.session_state.test_proba = test_proba
    
    # Display results if model is trained
    if 'predictor' in st.session_state:
        display_results()
    else:
        display_welcome()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Software Defect Prediction Demo | Research & Educational Use Only</p>
    </div>
    """, unsafe_allow_html=True)


def display_welcome():
    """Display welcome message and instructions."""
    
    st.markdown("""
    ## Welcome to the Software Defect Prediction Demo
    
    This interactive demo allows you to:
    
    ### 🎯 Features
    - **Generate synthetic datasets** with realistic software metrics
    - **Train multiple ML models** (Random Forest, XGBoost, LightGBM, Neural Networks, Ensemble)
    - **Evaluate model performance** with comprehensive metrics
    - **Explain predictions** using SHAP analysis
    - **Visualize results** with interactive plots
    
    ### 📊 Metrics Included
    - **Classification metrics**: Accuracy, Precision, Recall, F1-Score
    - **AUC metrics**: ROC-AUC, PR-AUC
    - **Business metrics**: Precision@K, Recall@80% Precision
    - **Error analysis**: False Positive/Negative rates
    
    ### 🔧 How to Use
    1. **Configure parameters** in the sidebar
    2. **Click "Generate & Train Model"** to start
    3. **Explore results** in the tabs below
    4. **Analyze predictions** and explanations
    
    ### ⚠️ Important Notes
    - This uses **synthetic data** for demonstration
    - Results are **not production-ready**
    - Always validate with **real code review**
    - Use for **research and education** only
    """)


def display_results():
    """Display model results and analysis."""
    
    # Get data from session state
    data = st.session_state.data
    split_data = st.session_state.split_data
    predictor = st.session_state.predictor
    test_pred = st.session_state.test_pred
    test_proba = st.session_state.test_proba
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Performance", "🔍 Predictions", "🧠 Explainability", "📋 Dataset"
    ])
    
    with tab1:
        display_overview(data, predictor)
    
    with tab2:
        display_performance(split_data, predictor, test_pred, test_proba)
    
    with tab3:
        display_predictions(split_data, test_pred, test_proba)
    
    with tab4:
        display_explainability(split_data, predictor)
    
    with tab5:
        display_dataset_info(data)


def display_overview(data, predictor):
    """Display overview metrics."""
    
    st.header("📊 Model Overview")
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(data['features']))
    
    with col2:
        st.metric("Defect Rate", f"{data['y'].mean():.1%}")
    
    with col3:
        st.metric("Features", len(data['feature_names']))
    
    with col4:
        st.metric("Model Type", predictor.model_type.title())
    
    # Feature distribution
    st.subheader("Feature Distribution")
    
    # Select features to plot
    feature_cols = st.multiselect(
        "Select features to visualize",
        data['feature_names'],
        default=data['feature_names'][:4]
    )
    
    if feature_cols:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=feature_cols[:4],
            specs=[[{"secondary_y": False}] * 2] * 2
        )
        
        for i, feature in enumerate(feature_cols[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Histogram(
                    x=data['features'][feature],
                    name=feature,
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Feature Distributions"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_performance(split_data, predictor, test_pred, test_proba):
    """Display model performance metrics."""
    
    st.header("📈 Model Performance")
    
    # Evaluate model
    evaluator = DefectPredictionEvaluator()
    results = evaluator.evaluate_model(
        predictor.model, 
        split_data['X_test'], 
        split_data['y_test']
    )
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['metrics'].accuracy:.3f}")
        st.metric("Precision", f"{results['metrics'].precision:.3f}")
    
    with col2:
        st.metric("Recall", f"{results['metrics'].recall:.3f}")
        st.metric("F1-Score", f"{results['metrics'].f1:.3f}")
    
    with col3:
        st.metric("AUC-ROC", f"{results['metrics'].auc_roc:.3f}")
        st.metric("AUC-PR", f"{results['metrics'].auc_pr:.3f}")
    
    with col4:
        st.metric("Precision@10", f"{results['metrics'].precision_at_10:.3f}")
        st.metric("Precision@20", f"{results['metrics'].precision_at_20:.3f}")
    
    # Cross-validation results
    st.subheader("Cross-Validation Results")
    cv_mean = results['cv_mean']
    cv_std = results['cv_std']
    
    st.metric(
        "CV AUC-ROC",
        f"{cv_mean:.3f} ± {cv_std:.3f}",
        delta=None
    )
    
    # ROC Curve
    st.subheader("ROC Curve")
    
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(split_data['y_test'], test_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=400
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(split_data['y_test'], test_proba[:, 1])
    avg_precision = average_precision_score(split_data['y_test'], test_proba[:, 1])
    
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AP = {avg_precision:.3f})',
        line=dict(color='green', width=2)
    ))
    
    fig_pr.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=600,
        height=400
    )
    
    st.plotly_chart(fig_pr, use_container_width=True)


def display_predictions(split_data, test_pred, test_proba):
    """Display prediction analysis."""
    
    st.header("🔍 Prediction Analysis")
    
    # Prediction distribution
    st.subheader("Prediction Distribution")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=test_proba[:, 1],
        nbinsx=30,
        name='Defect Probability',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribution of Defect Probabilities",
        xaxis_title="Defect Probability",
        yaxis_title="Count",
        width=600,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk predictions
    st.subheader("High-Risk Predictions")
    
    high_risk_threshold = st.slider("High-risk threshold", 0.5, 0.9, 0.7)
    
    high_risk_mask = test_proba[:, 1] > high_risk_threshold
    high_risk_count = np.sum(high_risk_mask)
    
    st.metric("High-risk predictions", high_risk_count)
    
    if high_risk_count > 0:
        # Show high-risk samples
        high_risk_data = split_data['X_test'][high_risk_mask].copy()
        high_risk_data['defect_probability'] = test_proba[high_risk_mask, 1]
        high_risk_data['actual_label'] = split_data['y_test'][high_risk_mask]
        
        st.dataframe(
            high_risk_data.sort_values('defect_probability', ascending=False),
            use_container_width=True
        )
    
    # Prediction accuracy by probability range
    st.subheader("Accuracy by Probability Range")
    
    # Create probability bins
    bins = np.linspace(0, 1, 11)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins)-1):
        mask = (test_proba[:, 1] >= bins[i]) & (test_proba[:, 1] < bins[i+1])
        if np.sum(mask) > 0:
            accuracy = np.mean(test_pred[mask] == split_data['y_test'][mask])
            bin_accuracies.append(accuracy)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    fig_bins = go.Figure()
    fig_bins.add_trace(go.Bar(
        x=bin_labels,
        y=bin_accuracies,
        name='Accuracy',
        text=bin_counts,
        textposition='auto'
    ))
    
    fig_bins.update_layout(
        title="Accuracy by Probability Range",
        xaxis_title="Probability Range",
        yaxis_title="Accuracy",
        width=800,
        height=400
    )
    
    st.plotly_chart(fig_bins, use_container_width=True)


def display_explainability(split_data, predictor):
    """Display model explainability."""
    
    st.header("🧠 Model Explainability")
    
    # Feature importance
    if hasattr(predictor.model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        feature_importance = predictor.model.feature_importances_
        feature_names = split_data['feature_names']
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df.tail(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Feature Importance'
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # SHAP analysis
    st.subheader("SHAP Analysis")
    
    if st.button("Generate SHAP Explanations"):
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = DefectPredictionExplainer(
                    predictor.model, 
                    split_data['feature_names']
                )
                
                # Generate SHAP values
                shap_results = explainer.explain_with_shap(split_data['X_test'])
                
                # Store in session state
                st.session_state.explainer = explainer
                st.session_state.shap_results = shap_results
                
                st.success("SHAP explanations generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating SHAP explanations: {str(e)}")
    
    # Display SHAP results if available
    if 'explainer' in st.session_state:
        explainer = st.session_state.explainer
        
        # SHAP feature importance
        st.subheader("SHAP Feature Importance")
        
        shap_importance = explainer.get_feature_importance_shap()
        
        fig_shap = px.bar(
            x=shap_importance.head(15).values,
            y=shap_importance.head(15).index,
            orientation='h',
            title='Top 15 Features by SHAP Importance'
        )
        
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Instance explanation
        st.subheader("Instance Explanation")
        
        instance_idx = st.number_input(
            "Select instance to explain",
            0, len(split_data['X_test'])-1, 0
        )
        
        explanation = explainer.explain_instance(split_data['X_test'], instance_idx)
        
        st.write(f"**Instance {instance_idx} Prediction:**")
        if isinstance(explanation['prediction'], np.ndarray):
            st.write(f"Defect probability: {explanation['prediction'][1]:.3f}")
        else:
            st.write(f"Predicted class: {explanation['prediction']}")
        
        # Top contributing features
        st.write("**Top Contributing Features:**")
        for feature, shap_value in explanation['feature_importance'][:10]:
            st.write(f"- {feature}: {shap_value:.4f}")


def display_dataset_info(data):
    """Display dataset information."""
    
    st.header("📋 Dataset Information")
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    
    stats_df = data['features'].describe()
    st.dataframe(stats_df, use_container_width=True)
    
    # Feature correlations
    st.subheader("Feature Correlations")
    
    corr_matrix = data['features'].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Defect distribution by features
    st.subheader("Defect Distribution by Features")
    
    feature_to_analyze = st.selectbox(
        "Select feature to analyze",
        data['feature_names']
    )
    
    if feature_to_analyze:
        # Create bins for continuous features
        if data['features'][feature_to_analyze].dtype in ['int64', 'float64']:
            bins = pd.cut(data['features'][feature_to_analyze], bins=5)
            defect_by_bin = data['features'].groupby(bins)['defect'].mean()
            
            fig_defect = px.bar(
                x=defect_by_bin.index.astype(str),
                y=defect_by_bin.values,
                title=f"Defect Rate by {feature_to_analyze}",
                labels={'x': feature_to_analyze, 'y': 'Defect Rate'}
            )
        else:
            defect_by_cat = data['features'].groupby(feature_to_analyze)['defect'].mean()
            
            fig_defect = px.bar(
                x=defect_by_cat.index,
                y=defect_by_cat.values,
                title=f"Defect Rate by {feature_to_analyze}",
                labels={'x': feature_to_analyze, 'y': 'Defect Rate'}
            )
        
        st.plotly_chart(fig_defect, use_container_width=True)
    
    # Raw data preview
    st.subheader("Raw Data Preview")
    
    st.dataframe(data['features'].head(20), use_container_width=True)
    
    # Download data
    csv = data['features'].to_csv(index=False)
    st.download_button(
        label="Download Dataset as CSV",
        data=csv,
        file_name="defect_prediction_dataset.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
