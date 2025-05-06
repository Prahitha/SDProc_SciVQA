from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="microsoft/Phi-4-multimodal-instruct")
print(model_path)

