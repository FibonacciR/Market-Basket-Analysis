import os
import torch
import pickle
from typing import Optional
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ============================================================
#  NCF MODEL
# ============================================================

def load_ncf_model():
    path = os.path.join(MODELS_DIR, 'best_ncf_model.pt')
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    from src.model import NCF

    num_users = 131209
    num_products = 47913

    model = NCF(num_users=num_users, num_products=num_products)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_ncf(model, user_id, basket, top_k: int = 10, num_products_default: int = 47913):
    """
    Score candidate products with the NCF model and return recommendations.

    Parameters:
        model: torch model (NCF)
        user_id: int
        basket: list[int] (product ids to exclude from recommendations)
        top_k: number of recommendations to return

    Returns:
        dict with keys:
        - input_scores: list of scores for the provided basket (aligned)
        - recommended: list of {product_id, score, probability, product_name}
        Or returns {'error': '...'} on failure.
    """
    try:
        if isinstance(model, dict):
            return {"error": "El modelo NCF cargado es un dict y no puede usarse para predicción. Verifica el archivo del modelo."}

        # Use model's actual product vocabulary (0 to num_products-1)
        candidate_ids = list(range(min(num_products_default, model.product_embedding.num_embeddings)))

        # Load product names for display
        try:
            from src.utils import load_product_names
            product_names = load_product_names()
        except Exception:
            product_names = {}

        # Compute scores for the input basket
        input_scores = []
        if basket:
            user_tensor = torch.tensor([user_id] * len(basket), dtype=torch.long)
            basket_tensor = torch.tensor(basket, dtype=torch.long)
            with torch.no_grad():
                out = model(user_tensor, basket_tensor)
                input_scores = [float(x) for x in out.tolist()]

        # Score all candidate products
        device = torch.device('cpu')
        model.to(device)
        model.eval()

        batch_size = 4096
        scores = []

        with torch.no_grad():
            for i in range(0, len(candidate_ids), batch_size):
                chunk = candidate_ids[i:i + batch_size]
                u = torch.tensor([user_id] * len(chunk), dtype=torch.long, device=device)
                p = torch.tensor(chunk, dtype=torch.long, device=device)
                out = model(u, p)
                scores.extend([float(x) for x in out.tolist()])

        # Convert raw scores to probabilities via softmax
        scores_arr = np.array(scores, dtype=float)
        if scores_arr.size:
            max_s = np.max(scores_arr)
            exp_s = np.exp(scores_arr - max_s)
            probs = exp_s / np.sum(exp_s)
        else:
            probs = np.array([])

        # Build recommendation list excluding items in basket
        basket_set = set(int(x) for x in (basket or []))
        recs = []
        for pid, sc, pr in zip(candidate_ids, scores_arr.tolist(), probs.tolist() if probs.size else [0.0] * len(candidate_ids)):
            if pid in basket_set:
                continue
            recs.append({
                'product_id': int(pid),
                'score': float(sc),
                'probability': float(pr),
                'product_name': product_names.get(pid, '')
            })

        # Sort by probability desc and take top_k
        recs_sorted = sorted(recs, key=lambda x: x['probability'], reverse=True)[:top_k]

        return {'input_scores': input_scores, 'recommended': recs_sorted}

    except Exception as e:
        return {"error": f"NCF prediction error: {str(e)}"}


# ============================================================
#  LIGHTGBM MODEL
# ============================================================

def load_lgb_model():
    path = os.path.join(MODELS_DIR, 'baseline_lgb_model.txt')
    print(f"Intentando cargar modelo LightGBM desde: {path}")
    print(f"¿Existe el archivo?: {os.path.exists(path)}")
    if not os.path.exists(path):
        print("Archivo de modelo LightGBM no encontrado.")
        return None
    try:
        import lightgbm as lgb
        print("Importación de LightGBM exitosa. Intentando cargar el modelo...")
        booster = lgb.Booster(model_file=path)
        print("Modelo LightGBM cargado correctamente.")
        return booster
    except Exception as e:
        print(f"Error cargando LightGBM: {e}")
        return None


import pandas as pd
from src.utils import load_product_names

# Cargar features en memoria global solo una vez
_user_features = None
_product_features = None
_user_product_features = None
_user_features_dict = None
_product_features_dict = None
_user_product_features_dict = None
_features_loaded = False

def _load_features():
    global _user_features, _product_features, _user_product_features, _features_loaded
    global _user_features_dict, _product_features_dict, _user_product_features_dict
    if not _features_loaded:
        _user_features = pd.read_pickle(os.path.join(MODELS_DIR, 'user_features.pkl'))
        _product_features = pd.read_pickle(os.path.join(MODELS_DIR, 'product_features.pkl'))
        _user_product_features = pd.read_pickle(os.path.join(MODELS_DIR, 'user_product_features.pkl'))
        # Convertir a diccionarios para acceso rápido
        _user_features_dict = {row['user_id']: row for _, row in _user_features.iterrows()}
        _product_features_dict = {row['product_id']: row for _, row in _product_features.iterrows()}
        _user_product_features_dict = {}
        for _, row in _user_product_features.iterrows():
            _user_product_features_dict[(row['user_id'], row['product_id'])] = row
        _features_loaded = True

def predict_lgb(model, user_id, basket):
    """
    Temporary simple LightGBM prediction - returns basic recommendations without heavy feature loading.
    """
    if model is None:
        return {"error": "LightGBM model not loaded."}
    try:
        print(f"Simple LGB prediction for user {user_id} with basket {basket}")
        
        from src.utils import load_product_names
        product_names = load_product_names()
        
        basket_set = set(basket or [])
        
        # Get candidates from product catalog dynamically (first 100 products excluding basket)
        import pandas as pd
        try:
            products_df = pd.read_csv(os.path.join(os.path.dirname(MODELS_DIR), 'data', 'products.csv'))
            all_product_ids = products_df['product_id'].tolist()
            candidate_ids = [pid for pid in all_product_ids if pid not in basket_set][:20]
        except:
            # Fallback if CSV not found
            candidate_ids = [i for i in range(1, 21) if i not in basket_set]
        print(f"Selected {len(candidate_ids)} candidates: {candidate_ids}")
        
        # Create simple feature vectors (using dummy values that match model expectations)
        feature_vectors = []
        meta = []
        
        for pid in candidate_ids:
            # Simple dummy features that match the 18 expected features
            fv = [
                1,    # frequency
                10,   # user_total_orders
                5,    # user_order_count
                2,    # user_favorite_dow
                14,   # user_avg_hour
                7,    # user_avg_days_between
                20,   # user_distinct_products
                100,  # user_total_products
                0.5,  # user_reorder_ratio
                8,    # user_avg_basket_size
                2,    # user_basket_std
                1000, # product_orders_count
                0.3,  # product_reorder_ratio
                5,    # product_avg_cart_position
                0.8,  # product_popularity_percentile
                3,    # up_orders_count
                0.4,  # up_reorder_ratio
                3     # up_avg_cart_position
            ]
            feature_vectors.append(fv)
            meta.append({
                'product_id': int(pid),
                'product_name': product_names.get(pid, f'Product {pid}'),
            })
        
        print(f"Built {len(feature_vectors)} feature vectors")
        if not feature_vectors:
            return {"error": "No candidate products found."}
        
        print("Making predictions...")
        features = np.array(feature_vectors)
        preds = model.predict(features)
        print(f"Generated {len(preds)} predictions")
        
        # Get top recommendations
        top_k = min(10, len(preds))
        top_indices = np.argsort(preds)[::-1][:top_k]
        
        recommended = []
        for idx in top_indices:
            recommended.append({
                'product_id': meta[idx]['product_id'],
                'score': float(preds[idx]),
                'probability': float(preds[idx]),
                'product_name': meta[idx]['product_name']
            })
        
        # Simple basket scores
        scores_aligned = []
        for pid in basket:
            pname = product_names.get(pid, f'Product {pid}')
            # Use a simple dummy prediction for basket items
            simple_features = [1, 10, 5, 2, 14, 7, 20, 100, 0.5, 8, 2, 500, 0.5, 3, 0.6, 5, 0.6, 2]
            score = float(model.predict(np.array([simple_features]))[0])
            scores_aligned.append({
                'product_id': int(pid),
                'score': score,
                'product_name': pname
            })
        
        recommended_ids = [r['product_id'] for r in recommended]
        print(f"Returning {len(recommended)} recommendations and {len(scores_aligned)} basket scores")
        
        return {
            'scores': scores_aligned,
            'recommended': recommended,
            'recommended_ids': recommended_ids
        }
        
        print("Step 4: Checking user features...")
        # Check if user exists
        if user_id not in _user_features_dict:
            return {"error": f"User features not found for user_id {user_id}"}
        uf = _user_features_dict[user_id]
        print(f"Step 4: User {user_id} features found")
        
        print("Step 5: Building feature vectors...")
        feature_names = [
            'frequency', 'user_total_orders', 'user_order_count', 'user_favorite_dow', 'user_avg_hour',
            'user_avg_days_between', 'user_distinct_products', 'user_total_products', 'user_reorder_ratio',
            'user_avg_basket_size', 'user_basket_std', 'product_orders_count', 'product_reorder_ratio',
            'product_avg_cart_position', 'product_popularity_percentile', 'up_orders_count', 'up_reorder_ratio', 'up_avg_cart_position'
        ]
        feature_vectors = []
        meta = []
        processed_count = 0
        for pid in candidate_ids:
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processing product {processed_count}/{len(candidate_ids)}")
            # frequency y user-product features
            up_row = _user_product_features_dict.get((user_id, pid), None)
            frequency = int(up_row['up_orders_count']) if up_row is not None else 0
            up_orders_count = int(up_row['up_orders_count']) if up_row is not None else 0
            up_reorder_ratio = float(up_row['up_reorder_ratio']) if up_row is not None else 0.0
            up_avg_cart_position = float(up_row['up_avg_cart_position']) if up_row is not None else 0.0
            # features del producto
            pf = _product_features_dict.get(pid, None)
            if pf is None:
                continue
            fv = [
                frequency,
                int(uf['user_total_orders']),
                int(uf['user_order_count']),
                int(uf['user_favorite_dow']),
                float(uf['user_avg_hour']),
                float(uf['user_avg_days_between']),
                int(uf['user_distinct_products']),
                int(uf['user_total_products']),
                float(uf['user_reorder_ratio']),
                float(uf['user_avg_basket_size']),
                float(uf['user_basket_std']),
                int(pf['product_orders_count']),
                float(pf['product_reorder_ratio']),
                float(pf['product_avg_cart_position']),
                float(pf['product_popularity_percentile']),
                up_orders_count,
                up_reorder_ratio,
                up_avg_cart_position
            ]
            feature_vectors.append(fv)
            meta.append({
                'product_id': int(pid),
                'product_name': pf['product_name'],
            })
        print(f"Step 5: Built {len(feature_vectors)} feature vectors")
        if not feature_vectors:
            return {"error": "No candidate products with features found."}
        
        print("Step 6: Making predictions...")
        features = np.array(feature_vectors)
        preds = model.predict(features)
        print(f"Step 6: Generated {len(preds)} predictions")
        
        print("Step 7: Selecting top recommendations...")
        # Tomar top_k recomendaciones
        top_k = 10
        top_indices = np.argsort(preds)[::-1][:top_k]
        recommended = []
        for idx in top_indices:
            recommended.append({
                'product_id': meta[idx]['product_id'],
                'score': float(preds[idx]),
                'probability': float(preds[idx]),
                'product_name': meta[idx]['product_name']
            })
        print("Step 8: Calculating basket scores...")
        # Scores para productos en la cesta
        scores_aligned = []
        for pid in basket:
            pf = _product_features_dict.get(pid, None)
            pname = pf['product_name'] if pf is not None else ''
            up_row = _user_product_features_dict.get((user_id, pid), None)
            frequency = int(up_row['up_orders_count']) if up_row is not None else 0
            up_orders_count = int(up_row['up_orders_count']) if up_row is not None else 0
            up_reorder_ratio = float(up_row['up_reorder_ratio']) if up_row is not None else 0.0
            up_avg_cart_position = float(up_row['up_avg_cart_position']) if up_row is not None else 0.0
            if pf is None:
                continue
            fv = [
                frequency,
                int(uf['user_total_orders']),
                int(uf['user_order_count']),
                int(uf['user_favorite_dow']),
                float(uf['user_avg_hour']),
                float(uf['user_avg_days_between']),
                int(uf['user_distinct_products']),
                int(uf['user_total_products']),
                float(uf['user_reorder_ratio']),
                float(uf['user_avg_basket_size']),
                float(uf['user_basket_std']),
                int(pf['product_orders_count']),
                float(pf['product_reorder_ratio']),
                float(pf['product_avg_cart_position']),
                float(pf['product_popularity_percentile']),
                up_orders_count,
                up_reorder_ratio,
                up_avg_cart_position
            ]
            score = float(model.predict(np.array([fv]))[0])
            scores_aligned.append({
                'product_id': int(pid),
                'score': score,
                'product_name': pname
            })
        recommended_ids = [r['product_id'] for r in recommended]
        print("Step 9: Returning results...")
        print(f"Final result: {len(scores_aligned)} basket scores, {len(recommended)} recommendations")
        return {
            'scores': scores_aligned,
            'recommended': recommended,
            'recommended_ids': recommended_ids
        }
    except Exception as e:
        return {"error": f"LightGBM prediction error: {str(e)}"}


# ============================================================
#  ASSOCIATION RULES
# ============================================================

def load_association_rules():
    path = os.path.join(MODELS_DIR, 'association_rules.pkl')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def predict_association(rules, user_id, basket):
    """
    Predict recommendations using association rules.
    """
    try:
        if hasattr(rules, 'recommend') and callable(rules.recommend):
            return rules.recommend(basket)

        recommendations = []
        for rule in rules:
            if isinstance(rule, dict) and 'antecedents' in rule and 'consequents' in rule:
                if set(rule['antecedents']).issubset(set(basket)):
                    recommendations.extend(rule['consequents'])
        return list(set(recommendations))[:3]
    except Exception as e:
        return {"error": f"Association prediction error: {str(e)}"}
