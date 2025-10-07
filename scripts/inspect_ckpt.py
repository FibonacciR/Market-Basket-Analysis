import torch, os, sys
path = os.path.join('models','best_ncf_model.pt')
print('checkpoint path exists:', os.path.exists(path))
if not os.path.exists(path):
    sys.exit(1)
ckpt = torch.load(path, map_location='cpu')
print('top-level keys:', list(ckpt.keys()))
if 'model_state_dict' in ckpt:
    st = ckpt['model_state_dict']
else:
    st = ckpt
print('num params in state_dict:', len(st))
for k,v in st.items():
    try:
        print(k, tuple(v.shape))
    except Exception as e:
        print(k, type(v))
