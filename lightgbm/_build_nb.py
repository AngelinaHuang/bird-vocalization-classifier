import json, copy

def md(t):
    return {"cell_type":"markdown","metadata":{},"source":t.splitlines(True)}

def code(t):
    return {"cell_type":"code","metadata":{},"source":t.splitlines(True),"outputs":[],"execution_count":None}

# Load main notebook to copy cell sources
with open("E:/MLwork/MLwork/bird-vocalization-classifier/YAMNet/src/yamnet_frozen_lightgbm_export.ipynb", encoding="utf-8") as f:
    mnb = json.load(f)

def msrc(idx):
    return "".join(mnb["cells"][idx]["source"])

# Get sources from main notebook
cfg_src = msrc(2)    # imports + Cfg
scan_fn_src = msrc(4)  # scan functions (no execution at bottom)
stream_src = msrc(6)  # streaming + YAMNet
cache_full_src = msrc(8)  # cache + export + cv head
noise_src = msrc(10)  # noise functions

print("Sources loaded from main notebook")
print(f"cfg: {len(cfg_src)} chars")
print(f"scan: {len(scan_fn_src)} chars")
print(f"stream: {len(stream_src)} chars")
print(f"cache: {len(cache_full_src)} chars")
print(f"noise: {len(noise_src)} chars")