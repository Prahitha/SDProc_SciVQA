from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Infyn/Bespoke-SFT-GRPO-700",
    repo_type="model",
    local_dir="Bespoke-SFT-GRPO-700",
    use_auth_token=True,  # Uses your token from `huggingface-cli login`
    resume_download=True,
    local_dir_use_symlinks=False
)

