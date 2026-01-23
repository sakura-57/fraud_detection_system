"""
Practice SQL equivalents of pandas operations used in feature engineering
"""
import pandas as pd
import sqlite3
from pathlib import Path
import os
import numpy as np

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load data
DATA_PATH = PROJECT_ROOT / 'data' / 'processed'

if (DATA_PATH / 'engineered_data.parquet').exists():
    df = pd.read_parquet(DATA_PATH / 'engineered_data.parquet')
    print(f"✅ Loaded data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:20]}...")  # Show first 20 columns
else:
    print(f"❌ File not found: {DATA_PATH / 'engineered_data.parquet'}")
    exit(1)

# Create SQLite database
conn = sqlite3.connect(':memory:')

# Register custom aggregate functions for SQLite
class StdDev:
    """Custom aggregate function for standard deviation in SQLite"""
    def __init__(self):
        self.values = []
    
    def step(self, value):
        if value is not None:
            self.values.append(value)
    
    def finalize(self):
        if len(self.values) < 2:
            return 0.0
        return np.std(self.values, ddof=0)  # Population std dev

# Register custom functions
conn.create_aggregate("STDDEV", 1, StdDev)

df.to_sql('transactions', conn, index=False, if_exists='replace')

# Check if Transaction_hour exists in columns
print(f"\nChecking for Transaction_hour column...")
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(transactions);")
columns = cursor.fetchall()
print(f"Table has {len(columns)} columns")
hour_exists = any('transaction_hour' in col[1].lower() for col in columns)
print(f"Transaction_hour exists: {hour_exists}")

# Sample some data to understand structure
cursor.execute("SELECT * FROM transactions LIMIT 5")
sample = cursor.fetchall()
print(f"\nSample row: {sample[0][:10]}...")  # First 10 values

# Practice SQL queries - updated with SQLite-compatible functions
queries = {
    # 1. Basic aggregations
    "fraud_rate_by_hour": """
        SELECT Transaction_hour, 
               AVG(isFraud) as fraud_rate,
               COUNT(*) as transaction_count
        FROM transactions
        WHERE Transaction_hour IS NOT NULL
        GROUP BY Transaction_hour
        ORDER BY Transaction_hour
    """,
    
    # 2. Top fraudulent cards
    "top_fraudulent_cards": """
        SELECT card1,
               AVG(isFraud) as fraud_rate,
               COUNT(*) as total_transactions,
               AVG(TransactionAmt) as avg_amount
        FROM transactions
        WHERE card1 IS NOT NULL
        GROUP BY card1
        HAVING COUNT(*) > 10
        ORDER BY fraud_rate DESC
        LIMIT 10
    """,
    
    # 3. User behavior aggregations with SQLite-compatible std dev
    "user_behavior_aggregations_sqlite": """
        WITH user_stats AS (
            SELECT card1,
                   COUNT(*) as transaction_count,
                   AVG(TransactionAmt) as avg_amount,
                   AVG(TransactionAmt * TransactionAmt) - AVG(TransactionAmt) * AVG(TransactionAmt) as variance,
                   SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count
            FROM transactions
            WHERE card1 IS NOT NULL
            GROUP BY card1
        )
        SELECT t.TransactionID,
               t.TransactionAmt,
               t.isFraud,
               us.transaction_count as card1_transaction_count,
               us.avg_amount as card1_avg_amount,
               CASE 
                   WHEN us.variance > 0 THEN SQRT(us.variance)
                   ELSE 0 
               END as card1_std_amount,
               (t.TransactionAmt - us.avg_amount) / 
                   CASE 
                       WHEN us.variance > 0 THEN SQRT(us.variance)
                       ELSE 1 
                   END as amount_zscore
        FROM transactions t
        LEFT JOIN user_stats us ON t.card1 = us.card1
        WHERE t.card1 IS NOT NULL
        LIMIT 100
    """,
    
    # 4. Alternative: Use custom STDDEV function
    "user_behavior_with_custom_stddev": """
        WITH user_stats AS (
            SELECT card1,
                   COUNT(*) as transaction_count,
                   AVG(TransactionAmt) as avg_amount,
                   STDDEV(TransactionAmt) as std_amount,
                   SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count
            FROM transactions
            WHERE card1 IS NOT NULL
            GROUP BY card1
        )
        SELECT t.TransactionID,
               t.TransactionAmt,
               t.isFraud,
               us.transaction_count as card1_transaction_count,
               us.avg_amount as card1_avg_amount,
               us.std_amount,
               (t.TransactionAmt - us.avg_amount) / 
                   CASE 
                       WHEN us.std_amount > 0 THEN us.std_amount
                       ELSE 1 
                   END as amount_zscore
        FROM transactions t
        LEFT JOIN user_stats us ON t.card1 = us.card1
        WHERE t.card1 IS NOT NULL
        LIMIT 100
    """,
    
    # 5. Simple aggregations without std dev
    "simple_user_stats": """
        SELECT 
            card1,
            COUNT(*) as transaction_count,
            AVG(isFraud) as fraud_rate,
            MIN(TransactionAmt) as min_amount,
            MAX(TransactionAmt) as max_amount,
            AVG(TransactionAmt) as avg_amount,
            SUM(TransactionAmt) as total_amount
        FROM transactions
        WHERE card1 IS NOT NULL
        GROUP BY card1
        HAVING COUNT(*) > 5
        ORDER BY fraud_rate DESC
        LIMIT 10
    """,
    
    # 6. Time-based patterns
    "time_patterns": """
        SELECT 
            strftime('%H', datetime(TransactionDT, 'unixepoch')) as hour,
            COUNT(*) as transaction_count,
            AVG(isFraud) as fraud_rate,
            AVG(TransactionAmt) as avg_amount
        FROM transactions
        WHERE TransactionDT IS NOT NULL
        GROUP BY strftime('%H', datetime(TransactionDT, 'unixepoch'))
        ORDER BY hour
    """,
    
    # 7. Product code analysis
    "product_analysis": """
        SELECT 
            ProductCD,
            COUNT(*) as transaction_count,
            AVG(isFraud) as fraud_rate,
            AVG(TransactionAmt) as avg_amount,
            SUM(TransactionAmt) as total_volume
        FROM transactions
        WHERE ProductCD IS NOT NULL
        GROUP BY ProductCD
        ORDER BY fraud_rate DESC
    """
}

# Execute and compare with pandas
print("\n=== SQL Practice ===\n")

# Try the first query to see if it works
for name, query in queries.items():
    print(f"\nQuery: {name}")
    print("-" * 50)
    
    try:
        # SQL result
        sql_result = pd.read_sql_query(query, conn)
        print(f"✅ SQL result shape: {sql_result.shape}")
        print(sql_result.head())
        
        # Compare with pandas for simple queries
        if name == "fraud_rate_by_hour":
            # Calculate same thing in pandas
            df_hour = df.copy()
            df_hour = df_hour[df_hour['Transaction_hour'].notna()]
            pandas_result = df_hour.groupby('Transaction_hour')['isFraud'].agg(
                fraud_rate='mean',
                transaction_count='count'
            ).reset_index()
            
            print("\nPandas equivalent result (first 5):")
            print(pandas_result.head())
            
            # Compare
            print("\nComparison (SQL vs Pandas):")
            for i in range(min(5, len(sql_result), len(pandas_result))):
                sql_rate = sql_result.loc[i, 'fraud_rate']
                pandas_rate = pandas_result.loc[i, 'fraud_rate']
                diff = abs(sql_rate - pandas_rate)
                print(f"Hour {sql_result.loc[i, 'Transaction_hour']}: "
                      f"SQL={sql_rate:.6f}, Pandas={pandas_rate:.6f}, Diff={diff:.6f}")
        
    except Exception as e:
        print(f"❌ Error executing query: {e}")
        print(f"Query: {query[:200]}...")  # Show first 200 chars
    
print("\n✅ SQL practice complete")

# Show database statistics
cursor.execute("""
    SELECT 
        COUNT(*) as total_rows,
        AVG(isFraud) as overall_fraud_rate,
        AVG(TransactionAmt) as overall_avg_amount
    FROM transactions
""")
stats = cursor.fetchone()
print(f"\n📊 Database Statistics:")
print(f"   Total rows: {stats[0]:,}")
print(f"   Overall fraud rate: {stats[1]:.4%}")
print(f"   Average transaction amount: ${stats[2]:.2f}")

# Close connection
conn.close()