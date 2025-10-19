import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class GoldPricePredictor:
    def __init__(self, model_dir="models_improved"):
        self.model_dir = model_dir
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """加载所有训练好的模型"""
        try:
            # 加载特征信息
            feature_info = joblib.load(f"{self.model_dir}/feature_info.pkl")
            self.all_features = feature_info['all_features']
            self.selected_features = feature_info['selected_features']
            
            # 加载scaler
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            
            # 加载特征选择器（如果有）
            try:
                self.selector = joblib.load(f"{self.model_dir}/feature_selector.pkl")
                self.use_feature_selection = True
            except:
                self.use_feature_selection = False
            
            # 加载模型
            # LightGBM
            self.models['lgb'] = lgb.Booster(model_file=f"{self.model_dir}/lgb_model.txt")
            # XGBoost
            self.models['xgb'] = xgb.Booster()
            self.models['xgb'].load_model(f"{self.model_dir}/xgb_model.json")
            # CatBoost
            self.models['cat'] = CatBoostClassifier()
            self.models['cat'].load_model(f"{self.model_dir}/cat_model.cbm")
            # Random Forest & Logistic Regression
            self.models['rf'] = joblib.load(f"{self.model_dir}/rf_model.pkl")
            self.models['lr'] = joblib.load(f"{self.model_dir}/lr_model.pkl")
            # LSTM
            self.models['lstm'] = tf.keras.models.load_model(f"{self.model_dir}/lstm_model.keras")
            
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def create_features(self, df):
        """为新的数据创建特征"""
        # 这里应该包含与训练时相同的特征工程代码
        # 简化版本，实际使用时需要完整复制训练时的特征工程代码
        return df[self.all_features]
    
    def predict(self, new_data):
        """预测新的数据"""
        # 特征工程
        X_new = self.create_features(new_data)
        
        # 数据缩放
        X_scaled = self.scaler.transform(X_new)
        
        # 特征选择
        if self.use_feature_selection:
            X_scaled = self.selector.transform(X_scaled)
        
        # 各模型预测
        predictions = {}
        probabilities = {}
        
        # LightGBM
        lgb_proba = self.models['lgb'].predict(X_scaled)
        probabilities['lgb'] = lgb_proba
        
        # XGBoost
        dmatrix = xgb.DMatrix(X_scaled)
        xgb_proba = self.models['xgb'].predict(dmatrix)
        probabilities['xgb'] = xgb_proba
        
        # CatBoost
        cat_proba = self.models['cat'].predict_proba(X_scaled)[:, 1]
        probabilities['cat'] = cat_proba
        
        # 其他模型...
        rf_proba = self.models['rf'].predict_proba(X_scaled)[:, 1]
        probabilities['rf'] = rf_proba
        
        lr_proba = self.models['lr'].predict_proba(X_scaled)[:, 1]
        probabilities['lr'] = lr_proba
        
        # 集成预测
        ensemble_weights = {
            'lgb': 0.3, 'cat': 0.3, 'xgb': 0.15,
            'rf': 0.15, 'lr': 0.05
        }
        
        ensemble_proba = np.zeros(len(X_scaled))
        for name, weight in ensemble_weights.items():
            ensemble_proba += weight * probabilities[name]
        
        # 生成最终预测
        final_predictions = (ensemble_proba >= 0.5).astype(int)
        
        result_df = pd.DataFrame({
            'probability': ensemble_proba,
            'prediction': final_predictions,
            'signal': np.where(final_predictions == 1, 'BUY', 'SELL')
        })
        
        return result_df

# 使用示例
if __name__ == "__main__":
    predictor = GoldPricePredictor()
    
    # 假设有新数据
    # new_data = pd.read_csv("new_gold_data.csv")
    # predictions = predictor.predict(new_data)
    # print(predictions)