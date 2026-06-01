import os


def stitchiq_base():
    """Resolve StitchIQ workspace root (env override, Kaggle, or local)."""
    override = os.environ.get("STITCHIQ_BASE")
    if override:
        return override.rstrip("/")
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/stitchiq"
    return os.path.expanduser("~/stitchiq")
