import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FraudFeatureEngineer:
    """Advanced feature engineering for fraud detection"""
    
    def __init__(self):
        self.aggregated_features = {}
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from TransactionDT"""
        df = df.copy()
        
        # TransactionDT is likely seconds from reference time
        # Create cyclical time features
        df['Transaction_hour'] = (df['TransactionDT'] // 3600) % 24
        df['Transaction_day'] = (df['TransactionDT'] // (3600 * 24)) % 7
        
        # Cyclical encoding for hour
        df['Transaction_hour_sin'] = np.sin(2 * np.pi * df['Transaction_hour'] / 24)
        df['Transaction_hour_cos'] = np.cos(2 * np.pi * df['Transaction_hour'] / 24)
        
        # Is weekend?
        df['is_weekend'] = df['Transaction_day'].isin([5, 6]).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['Transaction_hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['night', 'morning', 'afternoon', 'evening'])
        
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Create aggregated features for user behavior"""
        df = df.copy()
        
        # For each grouping, create aggregations
        for group_col in group_cols:
            if group_col in df.columns:
                group_name = f"group_{group_col}"
                
                # Transaction amount statistics
                agg_amount = df.groupby(group_col)['TransactionAmt'].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ]).add_prefix(f'{group_col}_TransactionAmt_')
                
                # Fraud rate per group
                fraud_rate = df.groupby(group_col)['isFraud'].mean().rename(f'{group_col}_fraud_rate')
                
                # Merge back
                df = df.merge(agg_amount, how='left', left_on=group_col, right_index=True)
                df = df.merge(fraud_rate, how='left', left_on=group_col, right_index=True)
                
                # Rolling features (simplified)
                df[f'{group_col}_amount_zscore'] = (
                    df['TransactionAmt'] - df[f'{group_col}_TransactionAmt_mean']
                ) / df[f'{group_col}_TransactionAmt_std'].replace(0, 1)
                
                self.aggregated_features[group_col] = list(agg_amount.columns) + [fraud_rate.name]
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important columns"""
        df = df.copy()
        
        # Card-related interactions
        if 'card1' in df.columns and 'TransactionAmt' in df.columns:
            df['card1_amount_interaction'] = df['card1'] * np.log1p(df['TransactionAmt'])
        
        # Email domain combinations
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
            df['email_domain_combo'] = df['P_emaildomain'].astype(str) + '_' + df['R_emaildomain'].astype(str)
        
        # Device and card interaction
        if 'DeviceType' in df.columns and 'card4' in df.columns:
            df['device_card_combo'] = df['DeviceType'].astype(str) + '_' + df['card4'].astype(str)
        
        return df
    
    def create_velocity_features(self, df: pd.DataFrame, time_window: int = 3) -> pd.DataFrame:
        """Create velocity features (transactions per time window)"""
        df = df.copy()
        
        # This is simplified - in reality you'd need proper time series
        # For demo, we'll create simulated velocity features
        velocity_cols = ['card1', 'card2', 'addr1', 'P_emaildomain']
        
        for col in velocity_cols:
            if col in df.columns:
                # Count transactions per unique value (simplified)
                transaction_counts = df[col].value_counts().to_dict()
                df[f'{col}_transaction_count'] = df[col].map(transaction_counts)
                
                # Normalized by time (simplified)
                if 'TransactionDT' in df.columns:
                    df[f'{col}_velocity'] = df[f'{col}_transaction_count'] / (
                        df['TransactionDT'].max() - df['TransactionDT'].min() + 1
                    )
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("Starting feature engineering...")
        
        # 1. Time features
        df = self.create_time_features(df)
        print("✓ Created time features")
        
        # 2. Group by important columns for aggregations
        group_cols = ['card1', 'card2', 'addr1', 'P_emaildomain', 'DeviceType']
        df = self.create_aggregated_features(df, [c for c in group_cols if c in df.columns])
        print("✓ Created aggregated features")
        
        # 3. Interaction features
        df = self.create_interaction_features(df)
        print("✓ Created interaction features")
        
        # 4. Velocity features
        df = self.create_velocity_features(df)
        print("✓ Created velocity features")
        
        # 5. Additional engineered features
        # Transaction amount transformations
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
        df['TransactionAmt_percentile'] = df['TransactionAmt'].rank(pct=True)
        
        # Ratio features
        if 'C1' in df.columns and 'C2' in df.columns:
            df['C1_C2_ratio'] = df['C1'] / (df['C2'].replace(0, 1))
        
        print(f"Final shape after engineering: {df.shape}")
        return df