"""Debug script to quickly test model loading and prediction functions without running the API.

Usage:
    python scripts/debug_predict.py
"""

from pathlib import Path
import sys
from pathlib import Path
# Ensure project root is on sys.path so `src` package can be imported
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import (
    load_ncf_model, predict_ncf,
    load_lgb_model, predict_lgb,
    load_association_rules, predict_association
)

TEST_INPUT = {"user_id": 123, "basket": [10, 25, 37]}


def main():
    print('TEST INPUT:', TEST_INPUT)

    print('\n-- NCF --')
    ncf = load_ncf_model()
    print('ncf loaded:', type(ncf))
    try:
        out = predict_ncf(ncf, TEST_INPUT['user_id'], TEST_INPUT['basket'])
        print('ncf predict:', out)
    except Exception as e:
        print('ncf predict exception:', e)

    print('\n-- LightGBM --')
    lgbm = load_lgb_model()
    print('lgb loaded:', type(lgbm))
    try:
        out = predict_lgb(lgbm, TEST_INPUT['user_id'], TEST_INPUT['basket'])
        print('lgb predict:', out)
    except Exception as e:
        print('lgb predict exception:', e)

    print('\n-- Association --')
    rules = load_association_rules()
    print('association loaded:', type(rules))
    try:
        out = predict_association(rules, TEST_INPUT['user_id'], TEST_INPUT['basket'])
        print('association predict:', out)
    except Exception as e:
        print('association predict exception:', e)


if __name__ == '__main__':
    main()
