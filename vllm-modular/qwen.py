from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenGVLab/InternVL3-8B",
    repo_type="model",
    local_dir="InternVL3-8B",
    use_auth_token=True,  # Uses your token from `huggingface-cli login`
    resume_download=True,
    local_dir_use_symlinks=False
)

