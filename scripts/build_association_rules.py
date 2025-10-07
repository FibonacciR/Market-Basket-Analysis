"""Build a simple association rules object from order_products__train.csv.

This creates a lightweight model saved to models/association_rules.pkl with a `recommend(basket)`
method that returns the top co-occurring products for the given basket.

Usage:
    python scripts/build_association_rules.py
"""
from pathlib import Path
import pickle
from collections import defaultdict, Counter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / 'order_products__train.csv'
OUTPUT_FILE = MODELS_DIR / 'association_rules.pkl'

class SimpleAssociationRules:
    def __init__(self, cooccurrence):
        # cooccurrence: dict[item_id] -> Counter({other_item: count})
        self.cooccurrence = cooccurrence

    def recommend(self, basket, top_k=3):
        if not basket:
            return []
        agg = Counter()
        for item in basket:
            agg.update(self.cooccurrence.get(int(item), Counter()))
        # remove items already in basket
        for b in basket:
            agg.pop(int(b), None)
        return [int(pid) for pid, _ in agg.most_common(top_k)]


def build():
    if not TRAIN_FILE.exists():
        print(f"Train file not found: {TRAIN_FILE}")
        return False

    print('Reading train data...')
    df = pd.read_csv(TRAIN_FILE)
    # group products per order_id
    orders = df.groupby('order_id')['product_id'].apply(list)

    cooccurrence = defaultdict(Counter)
    print('Computing co-occurrence counts...')
    for products in orders:
        unique = set(int(p) for p in products)
        for a in unique:
            for b in unique:
                if a == b:
                    continue
                cooccurrence[a][b] += 1

    model = SimpleAssociationRules(cooccurrence)

    print(f'Saving model to {OUTPUT_FILE} ...')
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(model, f)

    print('Done.')
    return True


if __name__ == '__main__':
    build()
