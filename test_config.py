from utils.config import Config

# Test the modified Config class
config = Config()
print(f"Default experiment name: {config.experiment_name}")
print(f"Expected format: image_guided_gvae_YYYYMMDD_HHMMSS")
print(f"Contains timestamp: {'_' in config.experiment_name and len(config.experiment_name.split('_')) == 3}")