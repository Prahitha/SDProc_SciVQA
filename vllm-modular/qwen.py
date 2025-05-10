from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Infyn/Qwen2.5-VL-7B-Instruct-SciVQA",
    repo_type="model",
    local_dir="Qwen2.5-VL-7B-Instruct-SciVQA",
    use_auth_token=True,  # Uses your token from `huggingface-cli login`
    resume_download=True,
    local_dir_use_symlinks=False
)

