import os, glob, torch
from safetensors import safe_open

def load_model(model: torch.nn.Module, path: str):
    params = dict(model.named_parameters())
    loaded = set()
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as fp:
            for name in fp.keys():
                if name == "lm_head.weight" and model.lm_head is None:
                    continue
                if name not in params:
                    raise KeyError(f"unexpected checkpoint key: {name}")
                p = params[name]
                w = fp.get_tensor(name)
                assert p.shape == w.shape, (name, p.shape, w.shape)
                p.data.copy_(w.to(p.dtype))
                loaded.add(name)
    missing = set(params) - loaded
    if missing:
        raise RuntimeError(f"uninitialised params: {sorted(missing)}")
