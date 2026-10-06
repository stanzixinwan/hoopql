"""HoopQL API. Query handling arrives in a later phase; this process only reports health."""

from fastapi import FastAPI

app = FastAPI(title="HoopQL", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
