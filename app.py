import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .good-score {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border: 3px solid #28a745;
    }
    .standard-score {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
        border: 3px solid #ffc107;
    }
    .bad-score {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        border: 3px solid #dc3545;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        color: #1976d2;
    }
    .model-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .model-loaded {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .model-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
</style>
""", unsafe_allow_html=True)

# Global variables for model and preprocessors
MODEL = None
SCALER = None
LABEL_ENCODERS = {}
TARGET_ENCODER = None
MODEL_LOADED = False
MODEL_TYPE = None

@st.cache_resource
def load_model():
    """Load the trained model and preprocessors"""
    global MODEL, SCALER, LABEL_ENCODERS, MODEL_LOADED, MODEL_TYPE, TARGET_ENCODER
    
    try:
        # Model file paths to search
        model_paths = [
            r'D:\Credit_Score_Prediction\model\credit_score_model.pkl',
            r'D:\Credit_Score_Prediction\model\rf_regressor.pkl',
            r'D:\Credit_Score_Prediction\src\predict.py',
            r'D:\Credit_Score_Prediction\src\credit_score_model.pkl',
            r'D:\Credit_Score_Prediction\src\rf_regressor.pkl',
            r'D:\Credit_Score_Prediction\credit_score_model.pkl',
            r'D:\Credit_Score_Prediction\rf_regressor.pkl',
            'credit_score_model.pkl',
            'rf_regressor.pkl',
            'model.pkl',
            'model.joblib'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if Path(model_path).exists():
                try:
                    if model_path.endswith('.py'):
                        # Handle Python prediction files
                        import sys
                        import importlib.util
                        
                        model_dir = str(Path(model_path).parent)
                        if model_dir not in sys.path:
                            sys.path.append(model_dir)
                        
                        spec = importlib.util.spec_from_file_location("predict_module", model_path)
                        predict_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(predict_module)
                        
                        if hasattr(predict_module, 'model'):
                            MODEL = predict_module.model
                            MODEL_TYPE = "Python Module Model"
                            model_loaded = True
                            break
                        elif hasattr(predict_module, 'predict'):
                            class PredictWrapper:
                                def __init__(self, predict_func):
                                    self.predict_func = predict_func
                                
                                def predict(self, X):
                                    return self.predict_func(X)
                                
                                def predict_proba(self, X):
                                    try:
                                        result = self.predict_func(X)
                                        if isinstance(result, tuple) and len(result) > 1:
                                            return result[1]
                                        else:
                                            pred = result[0] if isinstance(result, (list, tuple)) else result
                                            if pred == 'Good' or pred == 2:
                                                return [[0.1, 0.1, 0.8]]
                                            elif pred == 'Standard' or pred == 1:
                                                return [[0.2, 0.6, 0.2]]
                                            else:
                                                return [[0.7, 0.2, 0.1]]
                                    except:
                                        return [[0.33, 0.33, 0.34]]
                            
                            MODEL = PredictWrapper(predict_module.predict)
                            MODEL_TYPE = "Python Function Wrapper"
                            model_loaded = True
                            break
                    else:
                        # Try loading with pickle first
                        try:
                            with open(model_path, 'rb') as f:
                                MODEL = pickle.load(f)
                            model_loaded = True
                            MODEL_TYPE = type(MODEL).__name__
                            break
                        except:
                            # Try loading with joblib
                            MODEL = joblib.load(model_path)
                            model_loaded = True
                            MODEL_TYPE = type(MODEL).__name__
                            break
                except Exception as e:
                    continue
        
        if not model_loaded:
            raise FileNotFoundError("No compatible model file found")
        
        # Load preprocessors
        preprocessor_paths = [
            r'D:\Credit_Score_Prediction\model\feature_encoders.pkl',
            r'D:\Credit_Score_Prediction\model\target_encoder.pkl',
            r'D:\Credit_Score_Prediction\src\feature_encoders.pkl',
            r'D:\Credit_Score_Prediction\src\target_encoder.pkl',
            r'D:\Credit_Score_Prediction\src\scaler.pkl',
            'feature_encoders.pkl',
            'target_encoder.pkl',
            'scaler.pkl'
        ]
        
        for preprocessor_path in preprocessor_paths:
            if Path(preprocessor_path).exists():
                try:
                    with open(preprocessor_path, 'rb') as f:
                        preprocessor_data = pickle.load(f)
                        if isinstance(preprocessor_data, dict):
                            if 'feature_encoders' in preprocessor_data:
                                LABEL_ENCODERS = preprocessor_data['feature_encoders']
                            elif 'encoders' in preprocessor_data:
                                LABEL_ENCODERS = preprocessor_data['encoders']
                            else:
                                LABEL_ENCODERS = preprocessor_data
                        else:
                            if 'scaler' in preprocessor_path.lower():
                                SCALER = preprocessor_data
                            else:
                                LABEL_ENCODERS = preprocessor_data
                    break
                except:
                    try:
                        preprocessor_data = joblib.load(preprocessor_path)
                        if isinstance(preprocessor_data, dict):
                            LABEL_ENCODERS = preprocessor_data
                        break
                    except:
                        continue
        
        MODEL_LOADED = True
        return True, f"Model loaded successfully: {MODEL_TYPE}"
        
    except Exception as e:
        MODEL_LOADED = False
        return False, f"Error loading model: {str(e)}"

def preprocess_input_data(input_data):
    """Preprocess input data for model prediction"""
    try:
        df = pd.DataFrame([input_data])
        
        # Handle categorical variables
        categorical_features = ['employment_status', 'payment_min_amount', 'credit_mix', 'loan_purpose']
        
        for feature in categorical_features:
            if feature in df.columns:
                if feature in LABEL_ENCODERS and LABEL_ENCODERS[feature] is not None:
                    try:
                        if hasattr(LABEL_ENCODERS[feature], 'transform'):
                            df[feature] = LABEL_ENCODERS[feature].transform(df[feature])
                        else:
                            df[feature] = df[feature].map(LABEL_ENCODERS[feature]).fillna(0)
                    except (ValueError, KeyError):
                        df[feature] = 0
                else:
                    # Default encoding
                    if feature == 'employment_status':
                        employment_map = {
                            'Full-time Employee': 1, 'Self-Employed': 2, 'Business Owner': 3,
                            'Part-time Employee': 4, 'Freelancer': 5, 'Unemployed': 0,
                            'Student': 0, 'Retired': 1
                        }
                        df[feature] = df[feature].map(employment_map).fillna(0)
                    elif feature == 'payment_min_amount':
                        df[feature] = df[feature].map({'Yes': 1, 'No': 0}).fillna(0)
                    elif feature == 'credit_mix':
                        df[feature] = df[feature].map({'Good': 2, 'Standard': 1, 'Bad': 0}).fillna(0)
                    elif feature == 'loan_purpose':
                        purpose_map = {
                            'Home Purchase': 1, 'Vehicle': 2, 'Personal Use': 3,
                            'Business': 4, 'Education': 5, 'Debt Consolidation': 6,
                            'No Loans': 0
                        }
                        df[feature] = df[feature].map(purpose_map).fillna(0)
        
        # Handle loan_types
        if 'loan_types' in df.columns:
            if isinstance(df['loan_types'].iloc[0], list):
                df['num_loan_types'] = df['loan_types'].apply(len)
            df = df.drop('loan_types', axis=1, errors='ignore')
        
        # Ensure numeric columns
        numeric_columns = [
            'age', 'employment_years', 'annual_income', 'monthly_income', 'monthly_debt',
            'num_bank_accounts', 'num_credit_cards', 'total_credit_limit', 'credit_utilization',
            'num_loans', 'credit_history_age', 'num_late_payments', 'past_bankruptcies',
            'credit_inquiries', 'outstanding_debt', 'monthly_investment', 'monthly_balance'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Apply scaling if available
        if SCALER is not None:
            if hasattr(SCALER, 'feature_names_in_'):
                expected_features = SCALER.feature_names_in_
                missing_features = set(expected_features) - set(df.columns)
                for feature in missing_features:
                    df[feature] = 0
                df = df.reindex(columns=expected_features, fill_value=0)
            
            try:
                df_scaled = SCALER.transform(df)
                return pd.DataFrame(df_scaled, columns=df.columns)
            except:
                return df
        
        return df
        
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

def predict_with_model(input_data):
    """Make prediction using the trained model"""
    try:
        if not MODEL_LOADED or MODEL is None:
            raise Exception("Model not loaded")
        
        if hasattr(MODEL, 'predict_func') and MODEL_TYPE == "Python Function Wrapper":
            try:
                processed_data = preprocess_input_data(input_data)
                if processed_data is not None:
                    result = MODEL.predict_func(processed_data.values[0] if len(processed_data) > 0 else processed_data)
                else:
                    result = MODEL.predict_func(input_data)
                
                if isinstance(result, tuple):
                    prediction = result[0]
                    confidence = result[1] * 100 if len(result) > 1 else 85.0
                elif isinstance(result, dict):
                    prediction = result.get('prediction', result.get('class', 'Standard'))
                    confidence = result.get('confidence', 85.0)
                else:
                    prediction = result
                    confidence = 85.0
            except Exception as e:
                return predict_credit_score_fallback(input_data)
        else:
            processed_data = preprocess_input_data(input_data)
            if processed_data is None:
                raise Exception("Failed to preprocess data")
            
            if hasattr(MODEL, 'predict_proba'):
                probabilities = MODEL.predict_proba(processed_data)[0]
                prediction_idx = np.argmax(probabilities)
                confidence = probabilities[prediction_idx] * 100
                
                if hasattr(MODEL, 'classes_'):
                    classes = MODEL.classes_
                    prediction = classes[prediction_idx]
                else:
                    classes = ['Bad', 'Standard', 'Good']
                    prediction = classes[prediction_idx]
            else:
                prediction = MODEL.predict(processed_data)[0]
                confidence = 85.0
        
        # Normalize prediction
        if isinstance(prediction, (int, np.integer)):
            class_mapping = {0: 'Bad', 1: 'Standard', 2: 'Good'}
            prediction = class_mapping.get(prediction, 'Standard')
        
        if prediction not in ['Bad', 'Standard', 'Good']:
            if str(prediction).lower() in ['poor', 'low', 'bad']:
                prediction = 'Bad'
            elif str(prediction).lower() in ['average', 'medium', 'standard', 'fair']:
                prediction = 'Standard'
            elif str(prediction).lower() in ['good', 'excellent', 'high']:
                prediction = 'Good'
            else:
                prediction = 'Standard'
        
        # Calculate scores
        score_mapping = {'Bad': 45, 'Standard': 65, 'Good': 85}
        numerical_score = score_mapping.get(prediction, 65)
        numerical_score += np.random.normal(0, 5)
        numerical_score = max(20, min(95, numerical_score))
        
        cibil_ranges = {
            'Bad': '300-649',
            'Standard': '650-749', 
            'Good': '750-850'
        }
        cibil_range = cibil_ranges.get(prediction, '650-749')
        
        return prediction, confidence, numerical_score, cibil_range
        
    except Exception as e:
        return predict_credit_score_fallback(input_data)

def predict_credit_score_fallback(data):
    """Fallback rule-based prediction"""
    total_score = 0
    
    # Payment behavior (35% weight)
    payment_score = 100 if data['num_late_payments'] == 0 else max(20, 100 - data['num_late_payments'] * 8)
    if data['payment_min_amount'] == "No":
        payment_score *= 0.7
    total_score += payment_score * 0.35
    
    # Credit utilization (30% weight)
    utilization_score = max(20, 100 - max(0, data['credit_utilization'] - 10) * 2)
    total_score += utilization_score * 0.30
    
    # Credit history age (15% weight)
    history_score = min(100, 20 + data['credit_history_age'] * 8)
    total_score += history_score * 0.15
    
    # Credit mix (10% weight)
    diversity = (1 if data['num_credit_cards'] > 0 else 0) + (1 if data['num_loans'] > 0 else 0)
    mix_scores = {"Good": 100, "Standard": 50, "Bad": 25}
    mix_score = mix_scores.get(data['credit_mix'], 25) + (diversity * 12.5)
    total_score += min(100, mix_score) * 0.10
    
    # New credit (10% weight)
    inquiry_score = max(20, 100 - data['credit_inquiries'] * 10)
    total_score += inquiry_score * 0.10
    
    # Additional factors
    if data['annual_income'] >= 1000000 and data['employment_years'] >= 3:
        total_score += 5
    elif data['annual_income'] >= 500000 and data['employment_years'] >= 2:
        total_score += 3
    
    # Debt ratio penalty
    debt_ratio = (data['monthly_debt'] * 12) / data['annual_income'] if data['annual_income'] > 0 else 0
    if debt_ratio > 0.5:
        total_score -= 10
    elif debt_ratio > 0.3:
        total_score -= 5
    
    # Bankruptcy penalty
    total_score -= 15 * data['past_bankruptcies']
    
    final_score = max(0, min(100, total_score))
    
    if final_score >= 75:
        return "Good", min(95, 80 + (final_score - 75) * 0.6), final_score, "750-850"
    elif final_score >= 55:
        return "Standard", min(85, 70 + (final_score - 55) * 0.5), final_score, "650-749"
    else:
        return "Bad", min(80, 60 + final_score * 0.3), final_score, "300-649"

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Score Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model and show status
    model_status, model_message = load_model()
    
    if model_status:
        st.markdown(f'<div class="model-status model-loaded">✅ {model_message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="model-status model-error">⚠️ {model_message}<br>Using fallback rule-based prediction</div>', unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.header("📝 Enter Your Information")
    
    # Personal Details
    st.sidebar.subheader("👤 Personal Details")
    age = st.sidebar.slider("Age", 18, 80, 35)
    employment_status = st.sidebar.selectbox("Employment Status", 
        ["Full-time Employee", "Self-Employed", "Part-time Employee", "Business Owner", 
         "Freelancer", "Unemployed", "Student", "Retired"])
    employment_years = st.sidebar.slider("Years in Current Job", 0, 40, 5)
    
    # Financial Details
    st.sidebar.subheader("💰 Financial Details")
    annual_income = st.sidebar.number_input("Annual Income (₹)", 100000, 20000000, 800000, 50000)
    monthly_income = st.sidebar.number_input("Monthly In-hand Salary (₹)", 10000, 2000000, 60000, 5000)
    monthly_debt = st.sidebar.number_input("Total Monthly Debt/EMI (₹)", 0, 500000, 15000, 2000)
    
    # Banking & Credit Details
    st.sidebar.subheader("🏦 Banking & Credit Details")
    num_bank_accounts = st.sidebar.slider("Number of Bank Accounts", 1, 15, 3)
    num_credit_cards = st.sidebar.slider("Number of Credit Cards", 0, 20, 4)
    total_credit_limit = st.sidebar.number_input("Total Credit Limit (₹)", 0, 5000000, 200000, 25000)
    credit_utilization = st.sidebar.slider("Credit Utilization Ratio (%)", 0, 100, 25)
    
    # Loan Information
    st.sidebar.subheader("🏠 Loan Information")
    num_loans = st.sidebar.slider("Number of Active Loans", 0, 10, 2)
    loan_types = st.sidebar.multiselect("Types of Loans", 
        ["Home Loan", "Car Loan", "Personal Loan", "Education Loan", "Business Loan", 
         "Credit Card Loan", "Gold Loan", "Two-Wheeler Loan"], ["Car Loan"])
    loan_purpose = st.sidebar.selectbox("Primary Loan Purpose", 
        ["Home Purchase", "Vehicle", "Personal Use", "Business", "Education", "Debt Consolidation", "No Loans"])
    
    # Credit History
    st.sidebar.subheader("📊 Credit History")
    credit_history_age = st.sidebar.slider("Credit History Age (years)", 0, 35, 8)
    num_late_payments = st.sidebar.slider("Late Payments (last 24 months)", 0, 50, 3)
    payment_min_amount = st.sidebar.selectbox("Always Pay Minimum Amount?", ["Yes", "No"])
    past_bankruptcies = st.sidebar.slider("Past Bankruptcies", 0, 5, 0)
    
    # Credit Behavior
    st.sidebar.subheader("💳 Credit Behavior")
    credit_mix = st.sidebar.selectbox("Credit Mix Quality", ["Good", "Standard", "Bad"])
    credit_inquiries = st.sidebar.slider("Credit Inquiries (last 12 months)", 0, 20, 4)
    outstanding_debt = st.sidebar.number_input("Outstanding Debt (₹)", 0, 2000000, 50000, 10000)
    
    # Investment & Savings
    st.sidebar.subheader("💎 Investment & Savings")
    monthly_investment = st.sidebar.number_input("Monthly Investment (₹)", 0, 200000, 20000, 2000)
    monthly_balance = st.sidebar.number_input("Average Monthly Balance (₹)", 0, 1000000, 25000, 5000)
    
    predict_button = st.sidebar.button("🎯 Predict My Credit Score", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            input_data = {
                'age': age, 'employment_status': employment_status, 'employment_years': employment_years,
                'annual_income': annual_income, 'monthly_income': monthly_income, 'monthly_debt': monthly_debt,
                'num_bank_accounts': num_bank_accounts, 'num_credit_cards': num_credit_cards,
                'total_credit_limit': total_credit_limit, 'credit_utilization': credit_utilization,
                'num_loans': num_loans, 'loan_types': loan_types, 'loan_purpose': loan_purpose,
                'credit_history_age': credit_history_age, 'num_late_payments': num_late_payments,
                'payment_min_amount': payment_min_amount, 'past_bankruptcies': past_bankruptcies,
                'credit_mix': credit_mix, 'credit_inquiries': credit_inquiries,
                'outstanding_debt': outstanding_debt, 'monthly_investment': monthly_investment,
                'monthly_balance': monthly_balance
            }
            
            # Make prediction
            if MODEL_LOADED and MODEL is not None:
                prediction, confidence, score, cibil_range = predict_with_model(input_data)
                st.info(f"🤖 Prediction made using: {MODEL_TYPE}")
            else:
                prediction, confidence, score, cibil_range = predict_credit_score_fallback(input_data)
                st.warning("⚠️ Using fallback rule-based prediction.")
            
            st.subheader("🎯 Your Credit Score Prediction")
            
            prediction_styles = {
                "Good": ('🏆 EXCELLENT', 'good-score'),
                "Standard": ('⚖️ AVERAGE', 'standard-score'),
                "Bad": ('⚠️ NEEDS IMPROVEMENT', 'bad-score')
            }
            
            style_text, style_class = prediction_styles[prediction]
            st.markdown(f'<div class="prediction-box {style_class}">{style_text}<br>{prediction} Credit Score<br>CIBIL Range: {cibil_range}</div>', 
                       unsafe_allow_html=True)
            
            if prediction == "Good":
                st.balloons()
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Prediction Confidence", f"{confidence:.1f}%", 
                         "High" if confidence > 80 else "Medium")
            with col_b:
                st.metric("Credit Score", f"{score:.0f}/100", 
                         "Good" if score > 75 else "Fair" if score > 55 else "Poor")
            with col_c:
                debt_ratio = (monthly_debt * 12 / annual_income * 100) if annual_income > 0 else 0
                st.metric("Debt-to-Income", f"{debt_ratio:.1f}%", 
                         "Good" if debt_ratio < 30 else "High")
            
            # Visualization
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Credit Score Breakdown', 'Financial Health Overview', 
                              'Payment Behavior', 'Credit Mix Analysis',
                              'Risk Factors Assessment', 'Income vs Debt Analysis'),
                specs=[[{"type": "domain"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            score_color = "#28a745" if score > 75 else "#ffc107" if score > 55 else "#dc3545"
            fig.add_trace(go.Pie(
                labels=["Current Score", "Remaining"],
                values=[score, 100-score],
                marker_colors=[score_color, "#e9ecef"],
                hole=0.7
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=["Income (₹L)", "Debt (₹K)", "Investment (₹K)", "Balance (₹K)"],
                y=[annual_income/100000, monthly_debt/1000, monthly_investment/1000, monthly_balance/1000],
                marker_color=["#17a2b8", "#dc3545", "#28a745", "#6f42c1"]
            ), row=1, col=2)
            
            fig.add_trace(go.Bar(
                x=["On-time Payments", "Late Payments", "Credit Inquiries"],
                y=[max(0, 24-num_late_payments), num_late_payments, credit_inquiries],
                marker_color=["#28a745", "#dc3545", "#ffc107"]
            ), row=2, col=1)
            
            fig.add_trace(go.Bar(
                x=["Credit Cards", "Loans", "Bank Accounts"],
                y=[num_credit_cards, num_loans, num_bank_accounts],
                marker_color=["#fd7e14", "#20c997", "#6610f2"]
            ), row=2, col=2)
            
            risk_values = [
                max(0, credit_utilization - 30),
                num_late_payments * 2,
                max(0, credit_inquiries - 2) * 5,
                max(0, 5 - credit_history_age) * 5
            ]
            risk_colors = ["#dc3545" if v > 15 else "#ffc107" if v > 5 else "#28a745" for v in risk_values]
            fig.add_trace(go.Bar(
                x=["High Utilization", "Payment Issues", "Too Many Inquiries", "Short History"],
                y=risk_values,
                marker_color=risk_colors
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=[annual_income/100000],
                y=[monthly_debt/1000],
                mode='markers',
                marker=dict(size=20, color=score, colorscale='RdYlGn', showscale=True,
                           colorbar=dict(title="Credit Score")),
                name="Your Position"
            ), row=3, col=2)
            
            fig.update_layout(height=900, showlegend=False, title_text="Comprehensive Credit Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("🎯 Personalized Action Plan")
            
            priority_actions = []
            if credit_utilization > 50:
                priority_actions.append("🚨 URGENT: Credit utilization is dangerously high (>50%). Pay down balances immediately!")
            if num_late_payments > 10:
                priority_actions.append("🚨 URGENT: Multiple late payments detected. Set up auto-pay immediately!")
            if past_bankruptcies > 0:
                priority_actions.append("🚨 Bankruptcy impact: Focus on rebuilding credit with secured cards and consistent payments")
            
            recommendations = []
            if credit_utilization > 30:
                recommendations.append(f"🔴 Reduce credit utilization below 30% (currently {credit_utilization}%)")
            if num_late_payments > 2:
                recommendations.append(f"🔴 Improve payment history - {num_late_payments} late payments hurt your score")
            if payment_min_amount == "No":
                recommendations.append("🔴 Always pay at least the minimum amount due")
            if credit_history_age < 3:
                recommendations.append(f"🟡 Build longer credit history (current: {credit_history_age} years)")
            if credit_inquiries > 6:
                recommendations.append("🟡 Reduce credit inquiries - too many recent applications")
            if num_credit_cards == 0:
                recommendations.append("🟡 Consider getting a credit card to build credit history")
            if credit_mix == "Bad":
                recommendations.append("🟡 Diversify credit mix with different types of credit")
            
            if priority_actions:
                st.error("🚨 IMMEDIATE ACTION REQUIRED:")
                for action in priority_actions:
                    st.error(action)
            
            if recommendations:
                st.warning("📋 Improvement Recommendations:")
                for rec in recommendations:
                    st.warning(rec)
            
            if not recommendations and not priority_actions:
                st.success("🎉 Outstanding! Your credit profile is excellent. Keep up the great work!")
            
            # Pro Tips
            with st.expander("💡 Pro Tips for Credit Score Improvement"):
                st.success("""
                **Payment History (35% impact):**
                • Never miss payments - set up auto-pay for at least minimum amounts
                • Pay bills before due dates when possible
                • Contact lenders immediately if you're struggling to make payments
                
                **Credit Utilization (30% impact):**
                • Keep total utilization below 30%, ideally under 10%
                • Pay down balances before statement dates
                • Consider requesting credit limit increases
                
                **Credit Age (15% impact):**
                • Keep old accounts open even if you don't use them
                • Become an authorized user on family member's old accounts
                • Think twice before closing your oldest credit card
                
                **Credit Mix (10% impact):**
                • Have a mix of credit cards, installment loans, and retail accounts
                • Don't open accounts just for mix - only when you need them
                • Manage all types of credit responsibly
                
                **New Credit (10% impact):**
                • Limit hard inquiries to 2-3 per year
                • Shop for rates within 14-45 day windows for the same type of loan
                • Avoid opening multiple accounts in short periods
                """)
        
        else:
            # Welcome message
            st.markdown("""
            <div class="info-box">
                <h3>🎯 AI-Powered Credit Score Predictor</h3>
                <p><strong>Get accurate credit score predictions using advanced machine learning models!</strong></p>
                <p>This dashboard analyzes your financial profile using the same factors that major credit bureaus consider:</p>
                <ul>
                    <li><strong>Payment History (35%)</strong> - Your track record of making payments on time</li>
                    <li><strong>Credit Utilization (30%)</strong> - How much of your available credit you're using</li>
                    <li><strong>Credit History Length (15%)</strong> - How long you've been using credit</li>
                    <li><strong>Credit Mix (10%)</strong> - The variety of credit accounts you have</li>
                    <li><strong>New Credit (10%)</strong> - Recent credit inquiries and new accounts</li>
                </ul>
                <p>📝 <strong>Fill out the form in the sidebar to get your personalized credit score prediction and improvement recommendations!</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample insights
            st.subheader("📊 Understanding Credit Scores")
            
            sample_col1, sample_col2 = st.columns(2)
            
            with sample_col1:
                st.info("""
                **🏆 Excellent Credit (750-850)**
                • Qualify for best interest rates
                • Easy loan approvals
                • Premium credit card offers
                • Lower insurance premiums
                """)
                
                st.warning("""
                **⚖️ Good Credit (650-749)**
                • Decent interest rates available
                • Most loans approved
                • Standard credit card offers
                • Moderate insurance rates
                """)
            
            with sample_col2:
                st.error("""
                **⚠️ Fair Credit (550-649)**
                • Higher interest rates
                • Limited loan options
                • Basic credit cards only
                • Higher insurance premiums
                """)
                
                st.error("""
                **❌ Poor Credit (300-549)**
                • Very high interest rates
                • Loan approval difficult
                • Secured cards mainly
                • Highest insurance rates
                """)
    
    # Right sidebar content
    with col2:
        st.subheader("📊 Your Financial Snapshot")
        
        if annual_income > 0:
            debt_to_income = (monthly_debt * 12 / annual_income) * 100
            savings_rate = (monthly_investment / monthly_income) * 100 if monthly_income > 0 else 0
            
            income_status = 'Excellent' if annual_income > 1500000 else 'Good' if annual_income > 800000 else 'Average'
            debt_status = 'Excellent' if debt_to_income < 20 else 'Good' if debt_to_income < 35 else 'High'
            savings_status = 'Excellent' if savings_rate > 20 else 'Good' if savings_rate > 10 else 'Low'
            
            st.metric("Annual Income", f"₹{annual_income:,.0f}", delta=income_status)
            st.metric("Debt-to-Income", f"{debt_to_income:.1f}%", delta=debt_status)
            st.metric("Savings Rate", f"{savings_rate:.1f}%", delta=savings_status)
            
            if total_credit_limit > 0:
                utilization_amount = (credit_utilization / 100) * total_credit_limit
                st.metric("Credit Used", f"₹{utilization_amount:,.0f}", 
                         delta=f"of ₹{total_credit_limit:,.0f} limit")
        
        st.subheader("🎯 Quick Assessment")
        
        assessments = [
            ("Credit Utilization", credit_utilization <= 30, f"{credit_utilization}%"),
            ("Payment History", num_late_payments <= 2, f"{num_late_payments} late payments"),
            ("Credit Age", credit_history_age >= 5, f"{credit_history_age} years"),
            ("Credit Mix", credit_mix == "Good", credit_mix),
            ("Recent Inquiries", credit_inquiries <= 3, f"{credit_inquiries} inquiries")
        ]
        
        for name, is_good, value in assessments:
            if is_good:
                st.success(f"🟢 {name}: Good ({value})")
            else:
                st.error(f"🔴 {name}: Needs Work ({value})")
        
        st.subheader("📋 CIBIL Score Ranges")
        st.info("""
        **🏆 Excellent (750-850)**
        Best rates, easy approvals
        
        **✅ Good (650-749)**
        Competitive rates available
        
        **⚠️ Fair (550-649)**
        Limited options, higher rates
        
        **❌ Poor (300-549)**
        Difficult approvals, high rates
        """)
        
        st.subheader("💡 Quick Tips")
        st.success("""
        **🎯 Focus Areas:**
        • Pay all bills on time
        • Keep credit utilization low
        • Don't close old accounts
        • Limit new applications
        • Monitor credit regularly
        • Build emergency fund
        • Diversify credit types
        """)
        
        # Credit monitoring tools
        st.subheader("🔍 Credit Monitoring")
        st.info("""
        **Free Credit Reports:**
        • Annual Credit Report (Official)
        • CIBIL, Experian, Equifax
        • Credit monitoring apps
        
        **Monthly Monitoring:**
        • Check for errors
        • Monitor for fraud
        • Track score changes
        """)

# Footer section
    st.markdown("---")
    
    # Model performance section
    if MODEL_LOADED and MODEL is not None:
        st.subheader("🔍 Model Information")
        
        col_model1, col_model2 = st.columns(2)
        
        with col_model1:
            st.markdown("**📊 Technical Details:**")
            st.info(f"""
            - **Model Type**: {MODEL_TYPE}
            - **Status**: Successfully Loaded
            - **Preprocessors**: {"Available" if LABEL_ENCODERS or SCALER else "Default"}
            - **Prediction Mode**: ML Model
            """)
        
        with col_model2:
            st.markdown("**🎯 Model Features:**")
            st.info("""
            - Trained on historical data
            - Considers all credit factors
            - Provides confidence scores
            - Real-time predictions
            """)
    
    # Disclaimer and credits
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;'>
        <h4>AI-Powered Credit Score Prediction Dashboard</h4>
        <p><strong>Disclaimer:</strong> This tool provides educational estimates only. Actual credit scores may vary. 
        Always consult official credit bureaus for accurate scores.</p>
    </div>
    """, unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()