import os, glob, torch
from safetensors import safe_open

def load_model(model: torch.nn.Module, path: str):
    params = dict(model.named_parameters())
    loaded = set()
    
    merged_qkv = {}
    merged_gate_up = {}
    
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as fp:
            for name in fp.keys():
                if name == "lm_head.weight" and model.lm_head is None:
                    continue
                
                if ".q_proj." in name or ".k_proj." in name or ".v_proj." in name:
                    proj = "q_proj" if ".q_proj." in name else "k_proj" if ".k_proj." in name else "v_proj"
                    key = name.replace(f".{proj}.", ".qkv_proj.")
                    if key not in merged_qkv: merged_qkv[key] = {}
                    merged_qkv[key][proj] = fp.get_tensor(name)
                    continue
                    
                if ".gate_proj." in name or ".up_proj." in name:
                    proj = "gate_proj" if ".gate_proj." in name else "up_proj"
                    key = name.replace(f".{proj}.", ".gate_up_proj.")
                    if key not in merged_gate_up: merged_gate_up[key] = {}
                    merged_gate_up[key][proj] = fp.get_tensor(name)
                    continue
                    
                if name not in params:
                    raise KeyError(f"unexpected checkpoint key: {name}")
                p = params[name]
                w = fp.get_tensor(name)
                assert p.shape == w.shape, (name, p.shape, w.shape)
                p.data.copy_(w.to(p.dtype))
                loaded.add(name)
                
    for key, shards in merged_qkv.items():
        if key not in params: raise KeyError(f"missing target for merged key: {key}")
        w = torch.cat([shards["q_proj"], shards["k_proj"], shards["v_proj"]], dim=0)
        p = params[key]
        assert p.shape == w.shape, (key, p.shape, w.shape)
        p.data.copy_(w.to(p.dtype))
        loaded.add(key)
        
    for key, shards in merged_gate_up.items():
        if key not in params: raise KeyError(f"missing target for merged key: {key}")
        w = torch.cat([shards["gate_proj"], shards["up_proj"]], dim=0)
        p = params[key]
        assert p.shape == w.shape, (key, p.shape, w.shape)
        p.data.copy_(w.to(p.dtype))
        loaded.add(key)
        
    missing = set(params) - loaded
    if missing:
        raise RuntimeError(f"uninitialised params: {sorted(missing)}")
