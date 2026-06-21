from fastapi import FastAPI

app = FastAPI(
    title="DocFlow Agent",
    description="企业知识库与工单协同 Agent",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "DocFlow Agent API is running"
    }