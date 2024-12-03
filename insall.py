from huggingface_hub import snapshot_download

# Download the repository
repo_path = snapshot_download("RobbiePasquale/lightbulb")

print(f"Repository downloaded to: {repo_path}")
