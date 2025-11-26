from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='./models')
print("模型已下载到:", model_dir)