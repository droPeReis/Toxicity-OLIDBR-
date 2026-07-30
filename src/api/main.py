from fastapi import FastAPI
from .settings import Settings

args = Settings()

app = FastAPI(
    title=args.API_NAME,
    description=args.API_DESCRIPTION,
    version=args.API_VERSION,
)


@app.get(args.API_HEALTHCHECK_PATH)
def health():
    """Healthcheck endpoint"""
    return {"STATUS": "OK"}
