import os

from paths import stitchiq_base


def init_folders(base_path=None):
    if base_path is None:
        base_path = stitchiq_base()

    folders = [
        'raw_images/ankara',
        'raw_images/kente',
        'raw_images/asoke',
        'raw_images/lace',
        'raw_images/nigerian_styles',
        'clean_images',
        'captioned_dataset',
        'lora_output',
        'clip_index',
        'worker_models'
    ]

    print(f"Initializing StitchIQ workspace at: {base_path}")
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"  [Created/Exists] -> {path}")

    print("\nWorkspace folders are ready!")
    return base_path


if __name__ == "__main__":
    init_folders()
