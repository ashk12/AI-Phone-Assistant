from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import QueryRequest, QueryResponse
from app.core.phone_system import MultiIntentPhoneSystem
from app.config import LOCAL_PRODUCTS_JSON_PATH,GITHUB_PRODUCTS_JSON_PATH

router = APIRouter()


SYSTEM = MultiIntentPhoneSystem(GITHUB_PRODUCTS_JSON_PATH,LOCAL_PRODUCTS_JSON_PATH)

@router.post("/chat_stream")
async def query_phone_system_stream(req: QueryRequest):
    def generate():
        for chunk in SYSTEM.stream_response(req.query):
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")

@router.post("/chat", response_model=QueryResponse)
def query_phone_system(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    intent_data = SYSTEM.detect_intent(req.query)
    response_text = SYSTEM.process_query(req.query)

    return QueryResponse(
        intent=intent_data.get("intent", "recommendation"),
        confidence=intent_data.get("confidence", 0.7),
        response_text=response_text
    )

@router.get("/health")
def health_check():
    return {"status": "ok"}
