import pickle, os, sys
p='models/product_features.pkl'
print('exists', os.path.exists(p))
with open(p,'rb') as f:
    df=pickle.load(f)
print(type(df))
try:
    print('index sample:', list(df.index)[:10])
    print('columns:', df.columns.tolist()[:20])
except Exception as e:
    print('inspect error', e)
