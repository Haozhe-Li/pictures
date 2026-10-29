"""`python -m core.watermark`: fetch the TrustMark models (the Docker build runs this)."""
from .models import ensure_model, model_name

for part in ("encoder", "decoder"):
    print(ensure_model(model_name(part)))
