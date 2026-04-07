"""
generate_synthetic.py
---------------------
Generates a synthetic transactional sales dataset that mirrors the
structure of a real pricing dataset. The output is used to fit
a per-unit-price regression model and surface pricing leakage.

The data is designed so that the regression has real patterns to
recover (state freight differentials, volume discounting, product-
level price variation) alongside realistic confounders (ad-hoc
product x state anomalies, high-dispersion products, systematically
underpriced customers).

Run from the repo root:
    python3 scripts/generate_synthetic.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

# State multipliers reflect freight / remoteness premiums.
# NSW and VIC sit at the national baseline; WA, NT and TAS attract a
# meaningful premium; QLD and SA sit slightly above baseline.
STATE_MULTIPLIERS = {
    "NSW": 1.00,
    "VIC": 1.00,
    "QLD": 1.03,
    "SA": 1.04,
    "WA": 1.15,
    "NT": 1.18,
    "TAS": 1.12,
}

# Customer share by state — NSW and VIC dominate, NT and TAS are small.
STATE_CUSTOMER_WEIGHTS = {
    "NSW": 0.32,
    "VIC": 0.28,
    "QLD": 0.18,
    "SA": 0.09,
    "WA": 0.08,
    "NT": 0.02,
    "TAS": 0.03,
}

PRODUCT_CATEGORIES = ["Bakery", "Dairy", "Frozen", "Grocery", "Beverages"]

# Number of sub-categories per category. Sums to 27.
SUBCATS_PER_CATEGORY = {
    "Bakery": 5,
    "Dairy": 6,
    "Frozen": 5,
    "Grocery": 7,
    "Beverages": 4,
}

N_CUSTOMERS = 2000
N_PRODUCTS = 800
N_REPS = 30
PERIODS = ["Y0", "Y1"]
TARGET_TOTAL_ROWS = 100_000

# Share of products with wide price dispersion (negotiated specials etc.)
HIGH_DISPERSION_SHARE = 0.20

# Number of ad-hoc product x state anomalies to inject.
N_ADHOC_ANOMALIES = 50

# Customers that are systematically underpriced (the "leakage" cohort).
N_LEAKAGE_CUSTOMERS = 40

# Year-on-year inflation between Y0 and Y1.
INFLATION_Y1 = 0.04


# ---------------------------------------------------------------------------
# Build dimensions
# ---------------------------------------------------------------------------


def build_subcategories() -> list[tuple[str, str]]:
    """Return a list of (product_category, sub_category) tuples."""
    rows = []
    for cat, n in SUBCATS_PER_CATEGORY.items():
        for i in range(1, n + 1):
            rows.append((cat, f"{cat} Sub {i}"))
    return rows


def build_products(subcats: list[tuple[str, str]]) -> pd.DataFrame:
    """Build the product catalogue.

    Each product has a base national price drawn from a heavy-tailed
    distribution (most products are cheap, a long tail are premium),
    a sub-category (and therefore a category), and a flag indicating
    whether it is a high-dispersion product.
    """
    # Distribute 800 products across the 27 sub-categories proportionally.
    subcat_weights = RNG.dirichlet(np.ones(len(subcats)) * 2)
    counts = (subcat_weights * N_PRODUCTS).astype(int)
    counts[-1] = N_PRODUCTS - counts[:-1].sum()  # ensure we hit exactly 800

    rows = []
    pid = 1
    for (cat, subcat), n in zip(subcats, counts):
        for _ in range(n):
            # Log-normal base price gives most products in the $2-$30 range
            # with a long tail up to ~$100.
            base_price = float(np.clip(RNG.lognormal(mean=2.0, sigma=0.8), 1.0, 110.0))
            high_dispersion = RNG.random() < HIGH_DISPERSION_SHARE
            rows.append(
                {
                    "Product Description": f"Product {pid:04d}",
                    "Product Sub-Category": subcat,
                    "Product Category": cat,
                    "base_price": round(base_price, 2),
                    "high_dispersion": high_dispersion,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def build_reps() -> pd.DataFrame:
    """Build the sales rep roster. A handful are 'premium' reps."""
    rows = []
    for i in range(1, N_REPS + 1):
        rows.append(
            {
                "Rep": f"Rep {i:02d}",
                "premium": i <= 4,  # first 4 reps manage premium accounts
            }
        )
    return pd.DataFrame(rows)


def build_customers(reps: pd.DataFrame) -> pd.DataFrame:
    """Build the customer base.

    Each customer is assigned a home state, a sales rep, a size
    (Pareto-distributed so a few are huge), and a leakage flag.
    """
    states = list(STATE_CUSTOMER_WEIGHTS.keys())
    weights = list(STATE_CUSTOMER_WEIGHTS.values())

    customer_states = RNG.choice(states, size=N_CUSTOMERS, p=weights)

    # Pareto-distributed sizes — a few whales, a long tail of small accounts.
    sizes = RNG.pareto(a=1.5, size=N_CUSTOMERS) + 1
    sizes = sizes / sizes.mean()  # normalise so mean size = 1.0

    # Premium reps tend to manage the larger accounts.
    premium_reps = reps[reps["premium"]]["Rep"].tolist()
    standard_reps = reps[~reps["premium"]]["Rep"].tolist()

    # Sort customers by size descending, give the top ~15% to premium reps.
    size_order = np.argsort(-sizes)
    n_premium_customers = int(N_CUSTOMERS * 0.15)

    assigned_reps = np.empty(N_CUSTOMERS, dtype=object)
    for rank, idx in enumerate(size_order):
        if rank < n_premium_customers:
            assigned_reps[idx] = RNG.choice(premium_reps)
        else:
            assigned_reps[idx] = RNG.choice(standard_reps)

    # Pick leakage customers — bias toward larger ones so they show up
    # at the top of the leakage ranking later.
    leakage_probs = sizes / sizes.sum()
    leakage_idx = RNG.choice(
        N_CUSTOMERS, size=N_LEAKAGE_CUSTOMERS, replace=False, p=leakage_probs
    )
    leakage_flag = np.zeros(N_CUSTOMERS, dtype=bool)
    leakage_flag[leakage_idx] = True

    # Each leakage customer gets their own discount magnitude (10-25% off).
    leakage_discount = np.zeros(N_CUSTOMERS)
    leakage_discount[leakage_idx] = RNG.uniform(0.10, 0.25, size=N_LEAKAGE_CUSTOMERS)

    rows = []
    for i in range(N_CUSTOMERS):
        rows.append(
            {
                "Cust Code": f"C{i + 1:05d}",
                "Customer Name": f"Customer {i + 1:05d}",
                "State": customer_states[i],
                "Rep": assigned_reps[i],
                "size": sizes[i],
                "leakage_discount": leakage_discount[i],
            }
        )
    return pd.DataFrame(rows)


def build_adhoc_anomalies(products: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Pick random product x state combinations and assign weird multipliers."""
    anomalies = {}
    states = list(STATE_MULTIPLIERS.keys())
    product_ids = products["Product Description"].tolist()

    for _ in range(N_ADHOC_ANOMALIES):
        product = RNG.choice(product_ids)
        state = RNG.choice(states)
        # Multiplier well outside normal noise — 0.65 to 1.45.
        multiplier = float(RNG.choice([RNG.uniform(0.65, 0.80), RNG.uniform(1.25, 1.45)]))
        anomalies[(product, state)] = multiplier
    return anomalies


# ---------------------------------------------------------------------------
# Generate transactions
# ---------------------------------------------------------------------------


def generate_transactions(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    anomalies: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Generate the full transactional dataset."""
    product_lookup = products.set_index("Product Description").to_dict("index")
    product_ids = products["Product Description"].to_numpy()

    # Number of transactions per customer scales with size, split across periods.
    base_lines_per_customer = TARGET_TOTAL_ROWS / N_CUSTOMERS / len(PERIODS)
    sizes = customers["size"].to_numpy()
    sizes_norm = sizes / sizes.mean()
    lines_per_customer_per_period = np.maximum(
        1, np.round(base_lines_per_customer * sizes_norm).astype(int)
    )

    rows = []
    for i, customer in customers.iterrows():
        n_lines = lines_per_customer_per_period[i]

        for period in PERIODS:
            chosen = RNG.choice(product_ids, size=n_lines, replace=True)

            for product_id in chosen:
                product = product_lookup[product_id]
                base_price = product["base_price"]

                # Quantity — log-normal, scaled by customer size.
                qty = float(RNG.lognormal(mean=2.5, sigma=1.0)) * (0.5 + sizes[i])
                qty = max(1.0, round(qty, 2))

                state_mult = STATE_MULTIPLIERS[customer["State"]]

                # Volume discount — larger qty, lower per-unit price.
                volume_discount = min(0.30, 0.04 * np.log(qty))

                adhoc_mult = anomalies.get((product_id, customer["State"]), 1.0)
                leakage_mult = 1.0 - customer["leakage_discount"]

                if product["high_dispersion"]:
                    dispersion_noise = float(RNG.normal(1.0, 0.15))
                else:
                    dispersion_noise = float(RNG.normal(1.0, 0.04))
                dispersion_noise = max(0.5, dispersion_noise)

                period_mult = 1.0 + (INFLATION_Y1 if period == "Y1" else 0.0)

                price = (
                    base_price
                    * state_mult
                    * (1 - volume_discount)
                    * adhoc_mult
                    * leakage_mult
                    * dispersion_noise
                    * period_mult
                )
                price = max(0.50, round(price, 2))

                revenue = round(price * qty, 2)

                rows.append(
                    {
                        "Customer Name": customer["Customer Name"],
                        "Cust Code": customer["Cust Code"],
                        "State": customer["State"],
                        "Rep": customer["Rep"],
                        "Product Category": product["Product Category"],
                        "Product Sub-Category": product["Product Sub-Category"],
                        "Product Description": product_id,
                        "Period": period,
                        "Qty": qty,
                        "Revenue": revenue,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Building dimensions...")
    subcats = build_subcategories()
    products = build_products(subcats)
    reps = build_reps()
    customers = build_customers(reps)
    anomalies = build_adhoc_anomalies(products)

    print(f"  {len(products)} products across {len(subcats)} sub-categories")
    print(f"  {len(customers)} customers across {len(STATE_MULTIPLIERS)} states")
    print(f"  {len(reps)} reps")
    print(f"  {len(anomalies)} ad-hoc product x state anomalies")
    print(f"  {N_LEAKAGE_CUSTOMERS} leakage customers")

    print("Generating transactions...")
    transactions = generate_transactions(customers, products, anomalies)
    print(f"  {len(transactions):,} transaction rows generated")

    output_path = Path(__file__).resolve().parents[1] / "data" / "sales_synthetic.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()