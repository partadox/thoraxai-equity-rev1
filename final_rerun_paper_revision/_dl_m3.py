
import os
from huggingface_hub import snapshot_download
dst="/data/arta/thoraxai_research/1_equity/models/Qwen2.5-VL-3B-Instruct"
print("downloading to", dst, flush=True)
snapshot_download(repo_id="Qwen/Qwen2.5-VL-3B-Instruct", local_dir=dst,
                  ignore_patterns=["*.pth","original/*"])
print("DONE download", flush=True)
