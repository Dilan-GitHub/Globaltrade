import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. LOAD DATASETS
# ==========================================
tariffs_df = pd.read_csv("tariff_timeline.csv")
stocks_df = pd.read_csv("stock_market_reaction.csv")
sector_df = pd.read_csv("sector_impact.csv")
currency_df = pd.read_csv("currency_impact.csv")
inflation_df = pd.read_csv("inflation_response.csv")
trade_df = pd.read_csv("trade_volume_annual.csv")

# Ensure date formatting
for df in [
    tariffs_df,
    stocks_df,
    sector_df,
    currency_df,
    inflation_df,
]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

# ==========================================
# 2. EVENT CLASSIFICATION & TABLE 1: GLOBAL METRICS
# ==========================================
# Group tariff timeline events by date to classify event types
tariff_summary = (
    tariffs_df.groupby("date")
    .agg(
        has_initial=("retaliation", lambda x: (~x).any()),
        has_retaliation=("retaliation", "any"),
    )
    .reset_index()
)


def classify_event(row):
    if row["has_initial"] and row["has_retaliation"]:
        return "New Tariff + Retaliation (Same Day)"
    elif row["has_initial"] and not row["has_retaliation"]:
        return "Pure Initial Tariffs"
    elif not row["has_initial"] and row["has_retaliation"]:
        return "Pure Retaliatory Tariffs"
    else:
        return "Non-Event Baseline Days"


tariff_summary["event_category"] = tariff_summary.apply(classify_event, axis=1)

# Merge back into stock market reaction data
merged_stocks = pd.merge(
    stocks_df,
    tariff_summary[["date", "event_category"]],
    on="date",
    how="left",
)
merged_stocks["event_category"] = merged_stocks["event_category"].fillna(
    "Non-Event Baseline Days"
)


# Compute Table 1 Statistics
def compute_table1_stats(group):
    return pd.Series(
        {
            "Obs. Count": group["daily_return_pct"].count(),
            "Mean Daily Ret. (%)": group["daily_return_pct"].mean(),
            "Median Daily Ret. (%)": group["daily_return_pct"].median(),
            "Return Std. Dev. (%)": group["daily_return_pct"].std(),
            "Mean Weekly Ret. (%)": group["weekly_return_pct"].mean(),
            "20-Day Volatility": group["volatility_20d"].mean(),
        }
    )


table_1_results = merged_stocks.groupby("event_category").apply(
    compute_table1_stats
)
print("=== TABLE 1: GLOBAL EVENT CATEGORY METRICS ===")
print(table_1_results.round(3))


# ==========================================
# 3. REGIONAL MARKET IMPACT (TABLE 2)
# ==========================================
def map_region(country):
    if country == "USA":
        return "United States"
    elif country in ["Germany", "France", "UK", "Europe"]:
        return "Europe"
    elif country in ["China", "Hong Kong", "CHN"]:
        return "China & Hong Kong"
    return "Other"


stocks_df["region"] = stocks_df["country"].apply(map_region)
regional_merged = pd.merge(
    stocks_df,
    tariffs_df[["date", "retaliation"]].drop_duplicates(),
    on="date",
    how="left",
)
regional_merged["tariff_type"] = regional_merged["retaliation"].map(
    {False: "Pure Initial Tariff", True: "Pure Retaliation"}
)

table_2_results = (
    regional_merged.dropna(subset=["tariff_type"])
    .groupby(["region", "tariff_type"])
    .agg(
        Obs_Count=("daily_return_pct", "count"),
        Mean_Daily_Ret=("daily_return_pct", "mean"),
        Std_Deviation=("daily_return_pct", "std"),
        Volatility_20d=("volatility_20d", "mean"),
    )
)

print("\n=== TABLE 2: REGIONAL MARKET IMPACT ===")
print(table_2_results.round(3))


# ==========================================
# 4. SECTOR SENSITIVITY & ROTATION ANALYSIS
# ==========================================
# Analyze how high vs. low tariff sensitivity sectors perform near trade events
sector_merged = pd.merge(
    sector_df,
    tariffs_df[["date", "retaliation"]].drop_duplicates(),
    on="date",
    how="left",
)
sector_merged["event_type"] = sector_merged["retaliation"].map(
    {False: "Initial Tariff Date", True: "Retaliation Date"}
)

sector_performance = (
    sector_merged.dropna(subset=["event_type"])
    .groupby(["tariff_sensitivity", "sector_label", "event_type"])["daily_return_pct"]
    .mean()
    .unstack()
)

print("\n=== SECTOR PERFORMANCE ACROSS TARIFF EVENTS ===")
print(sector_performance.round(4))


# ==========================================
# 5. CURRENCY & INFLATION RESPONSE PIPELINE
# ==========================================
# Currency volatility around tariff events
currency_vol = (
    currency_df.groupby("tariff_event_nearby")["rolling_7d_vol"]
    .mean()
    .reset_index()
)
print("\n=== CURRENCY VOLATILITY (NEARBY TARIFF EVENTS) ===")
print(currency_vol)

# Inflation shift (Pre vs Post Trade War era)
inflation_summary = (
    inflation_df.groupby(["country", "post_tariff_era"])["yoy_change_pct"]
    .mean()
    .unstack()
)
print("\n=== INFLATION YOY CHANGE BY ERA ===")
print(inflation_summary.round(3))


# ==========================================
# 6. VISUALIZATION GENERATION (VOLATILITY COMPARISON)
# ==========================================
plt.figure(figsize=(10, 6))
categories = table_1_results.index
std_devs = table_1_results["Return Std. Dev. (%)"]

bars = plt.bar(
    categories,
    std_devs,
    color=["#2b5c8f", "#d9534f", "#f0ad4e", "#5bc0de"],
    edgecolor="black",
)
plt.title(
    "Return Standard Deviation Across Tariff Event Categories",
    fontsize=14,
    fontweight="bold",
)
plt.ylabel("Return Standard Deviation (%)", fontsize=12)
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.05,
        f"{yval:.3f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("tariff_volatility_comparison.png", dpi=300)
plt.close()
print(
    "\n[Success] Visualization saved successfully as 'tariff_volatility_comparison.png'."
)
