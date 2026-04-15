import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

REGIONS    = ["North", "South", "East", "West", "Central"]
SEGMENTS   = ["Premium", "Standard", "Basic"]
HOUSING    = ["Owner", "Renter", "Other"]

def generate_dataset(n=N):
    age               = np.random.randint(18, 75, n)
    gender            = np.random.choice([0, 1], n)
    region            = np.random.choice(REGIONS, n)
    segment           = np.random.choice(SEGMENTS, n)
    housing_type      = np.random.choice(HOUSING, n)
    has_car           = np.random.choice([0, 1], n, p=[0.4, 0.6])
    income_index      = np.random.randint(1, 6, n)
    product_holdings  = np.random.randint(1, 8, n)
    tenure_months     = np.random.randint(1, 120, n)
    digital_score     = np.random.randint(1, 5, n)
    activity_score    = np.random.randint(1, 5, n)
    engagement_level  = np.random.randint(1, 5, n)
    campaign_contacts = np.random.randint(0, 10, n)

    # Target: propensity to buy — higher income, more products, longer tenure = more likely
    target_prob = (
        0.05
        + (income_index / 5)     * 0.20
        + (product_holdings / 8) * 0.15
        + (tenure_months / 120)  * 0.10
        + (digital_score / 4)    * 0.10
        - (age / 75)             * 0.05
        + np.random.normal(0, 0.05, n)
    )
    target_prob = np.clip(target_prob, 0.02, 0.95)
    target      = (np.random.random(n) < target_prob).astype(int)

    df = pd.DataFrame({
        "customer_id":      [f"CUST_{i:05d}" for i in range(n)],
        "age":              age,
        "gender":           gender,
        "region":           region,
        "segment":          segment,
        "housing_type":     housing_type,
        "has_car":          has_car,
        "income_index":     income_index,
        "product_holdings": product_holdings,
        "tenure_months":    tenure_months,
        "digital_score":    digital_score,
        "activity_score":   activity_score,
        "engagement_level": engagement_level,
        "campaign_contacts":campaign_contacts,
        "target":           target,
    })
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_parquet("customer_data.parquet", index=False)
    print(f"Generated {len(df)} customer records.")
    print(f"Target rate: {df['target'].mean():.1%}")
    print(df.head())