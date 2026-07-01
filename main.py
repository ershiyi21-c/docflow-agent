# 标准库：JSON 用于向量序列化，Path/UUID 用于管理上传文件。
import json
from embedding import embed_text, embed_texts
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import Document, DocumentChunk, DocumentChunkEmbedding, Ticket
from llm import (
    classify_agent_intent,
    generate_knowledge_answer,
    generate_ticket_draft,
)
from typing import Literal
from pathlib import Path
from uuid import uuid4

# 根据 ORM 模型创建尚不存在的数据表。
Base.metadata.create_all(bind=engine)

# FastAPI 应用实例及其 OpenAPI 文档信息。
app = FastAPI(
    title="DocFlow Agent",
    description="企业知识库与工单协同 Agent",
    version="0.1.0",
)

# 上传目录、允许的文件类型，以及知识问答采用的最低相似度阈值。
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".md"}
MIN_SEMANTIC_SCORE = 0.55

# 将文档按固定字符数切分，供后续向量化和分片检索使用。
def split_text(text: str, chunk_size: int = 200) -> list[str]:
    text = text.strip()

    return [
        text[index:index + chunk_size]
        for index in range(0, len(text), chunk_size)
    ]

# 以下 Pydantic 模型负责校验接口接收的请求体。
class TicketCreate(BaseModel):
    title: str
    content: str
    priority: str = "normal"

class TicketStatusUpdate(BaseModel):
    status: Literal["open", "processing", "closed"]

class KnowledgeAsk(BaseModel):
    question: str

class AgentChatRequest(BaseModel):
    message: str

class TicketConfirmRequest(BaseModel):
    title: str
    content: str
    priority: Literal["high", "normal", "low"] = "normal"

    
# FastAPI 数据库依赖：每次请求创建会话，请求结束后确保关闭。
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# 健康检查接口，用于确认 API 服务是否正常运行。
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "DocFlow Agent API is running",
    }


# 创建工单，并在提交后刷新对象以取得数据库生成的字段。
@app.post("/tickets")
def create_ticket(ticket: TicketCreate, db: Session = Depends(get_db)):
    new_ticket = Ticket(**ticket.model_dump())

    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)

    return {
        "message": "工单创建成功",
        "ticket": new_ticket,
    }


# 按 ID 倒序返回全部工单，最新工单排在最前面。
@app.get("/tickets")
def list_tickets(db: Session = Depends(get_db)):
    ticket_items = db.query(Ticket).order_by(Ticket.id.desc()).all()

    return {
        "total": len(ticket_items),
        "items": ticket_items,
    }

# 根据主键查询单个工单，不存在时返回 404。
@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="工单不存在",
        )

    return ticket

# 更新指定工单的状态；允许值由 TicketStatusUpdate 限定。
@app.patch("/tickets/{ticket_id}/status")
def update_ticket_status(
    ticket_id: int,
    ticket_data: TicketStatusUpdate,
    db: Session = Depends(get_db),
):
    ticket = db.get(Ticket, ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="工单不存在",
        )

    ticket.status = ticket_data.status

    db.commit()
    db.refresh(ticket)

    return {
        "message": "工单状态更新成功",
        "ticket": ticket,
    }

# 接收文本类文档，完成保存、分片、向量化和向量入库。
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 文件名为空时无法判断文件类型，也无法记录原始名称。
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="请选择要上传的文件",
        )

    file_extension = Path(file.filename).suffix.lower()

    # 当前知识库仅处理可直接按 UTF-8 解码的纯文本文件。
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="当前仅支持 .txt 和 .md 文件",
        )

    saved_filename = f"{uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / saved_filename

    content = await file.read()
    file_path.write_bytes(content)

    # 在生成数据库记录前验证文本编码。
    try:
       text_content = content.decode("utf-8")
    except UnicodeDecodeError:
       raise HTTPException(
        status_code=400,
        detail="文件编码必须是 UTF-8",
    )

    # 保存文档元数据；实际文件内容已经写入 uploads 目录。
    document = Document(
        original_filename=file.filename,
        saved_filename=saved_filename,
        file_size=len(content),
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    # 将全文切分，并逐片保存到 document_chunks 表。
    chunks = split_text(text_content)

    chunk_objects = []

    for index, chunk_content in enumerate(chunks):
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            content=chunk_content,
        )
        chunk_objects.append(chunk)
        db.add(chunk)

    db.commit()

    # 批量生成归一化向量，顺序与 chunk_objects 保持一致。
    vectors = embed_texts(
        [chunk.content for chunk in chunk_objects]
    )

    for chunk, vector in zip(chunk_objects, vectors):
        # SQLite 不提供项目所需的向量字段，因此暂以 JSON 文本保存。
        embedding = DocumentChunkEmbedding(
            chunk_id=chunk.id,
            vector_json=json.dumps(vector),
        )
        db.add(embedding)

    db.commit()

    return {
        "message": "文档上传成功",
        "document": document,
    }

# 返回全部文档的元数据，最新上传的文档排在最前面。
@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.id.desc()).all()

    return {
        "total": len(documents),
        "items": documents,
    }

# 查询指定文档的所有分片，并按原始分片顺序返回。
@app.get("/documents/{document_id}/chunks")
def list_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="文档不存在",
        )

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )

    return {
        "document_id": document_id,
        "total": len(chunks),
        "items": chunks,
    }

# 使用 SQL contains 执行字面关键词检索，不涉及向量相似度。
@app.get("/documents/search")
def search_document_chunks(
    query: str,
    db: Session = Depends(get_db),
):
    keyword = query.strip()

    if not keyword:
        raise HTTPException(
            status_code=400,
            detail="查询关键词不能为空",
        )

    results = (
        db.query(DocumentChunk, Document.original_filename)
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .filter(DocumentChunk.content.contains(keyword))
        .order_by(
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
        )
        .all()
    )

    items = [
        {
            "document_id": chunk.document_id,
            "document_name": original_filename,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
        }
        for chunk, original_filename in results
    ]

    return {
        "query": keyword,
        "total": len(items),
        "items": items,
    }

# 检索最相关的知识片段，并让大语言模型基于这些资料回答。
@app.post("/knowledge/ask")
def ask_knowledge_base(
    request: KnowledgeAsk,
    db: Session = Depends(get_db),
):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空",
        )

    # 1. 将用户问题转换为向量
    query_vector = embed_text(question)

    # 2. 读取所有已保存向量的文档片段
    vector_results = (
        db.query(
            DocumentChunk,
            DocumentChunkEmbedding,
            Document.original_filename,
        )
        .join(
            DocumentChunkEmbedding,
            DocumentChunkEmbedding.chunk_id == DocumentChunk.id,
        )
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .all()
    )

    # 3. 计算问题向量与每个文档片段向量的相似度
    scored_results = []

    for chunk, embedding, original_filename in vector_results:
        # 入库时向量被序列化为 JSON，此处恢复为浮点数列表。
        chunk_vector = json.loads(embedding.vector_json)

        # 两侧向量均已归一化，因此点积可作为余弦相似度。
        score = sum(
            a * b
            for a, b in zip(query_vector, chunk_vector)
        )

        scored_results.append(
            (score, chunk, original_filename)
        )

    # 4. 相似度从高到低排序，取前三条
    scored_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top_results = scored_results[:3]

    # 没有结果或最高分过低时，不调用模型，避免无依据回答。
    if (
        not top_results
        or top_results[0][0] < MIN_SEMANTIC_SCORE
    ):
        best_score = (
            round(top_results[0][0], 4)
            if top_results
            else None
        )

        return {
            "answer": "知识库中没有足够相关的资料。",
            "best_score": best_score,
            "sources": [],
        }

    # 5. 组装检索到的资料，交给 DeepSeek 回答
    context_parts = []
    sources = []

    for score, chunk, original_filename in top_results:
        context_parts.append(
            f"文档：{original_filename}\n"
            f"内容：{chunk.content}"
        )

        sources.append(
            {
                "document_id": chunk.document_id,
                "document_name": original_filename,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 4),
            }
        )

    answer = generate_knowledge_answer(
        question=question,
        context="\n\n---\n\n".join(context_parts),
    )

    return {
        "answer": answer,
        "sources": sources,
    }

# 单独提供向量语义检索能力，返回相似度最高的 top_k 个分片。
@app.get("/documents/semantic-search")
def semantic_search_documents(
    query: str,
    top_k: int = 3,
    db: Session = Depends(get_db),
):
    question = query.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="查询内容不能为空",
        )

    query_vector = embed_text(question)

    results = (
        db.query(
            DocumentChunk,
            DocumentChunkEmbedding,
            Document.original_filename,
        )
        .join(
            DocumentChunkEmbedding,
            DocumentChunkEmbedding.chunk_id == DocumentChunk.id,
        )
        .join(
            Document,
            DocumentChunk.document_id == Document.id,
        )
        .all()
    )

    items = []

    for chunk, embedding, original_filename in results:
        chunk_vector = json.loads(embedding.vector_json)

        score = sum(
            a * b
            for a, b in zip(query_vector, chunk_vector)
        )

        items.append(
            {
                "document_id": chunk.document_id,
                "document_name": original_filename,
                "chunk_index": chunk.chunk_index,
                "score": round(score, 4),
                "content": chunk.content,
            }
        )

    # 先按相似度降序排列，再按调用方指定数量截取结果。
    items.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "query": question,
        "total": len(items[:top_k]),
        "items": items[:top_k],
    }
@app.post("/agent/chat")
def agent_chat(
    request: AgentChatRequest,
    db: Session = Depends(get_db),
):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="消息不能为空",
        )

    result = classify_agent_intent(message)
    intent = result["intent"]

    if intent == "knowledge":
        knowledge_result = ask_knowledge_base(
            KnowledgeAsk(question=message),
            db,
        )

        return {
            "intent": "knowledge",
            "result": knowledge_result,
        }

    if intent == "ticket_list":
        tickets = db.query(Ticket).order_by(Ticket.id.desc()).all()

        return {
            "intent": "ticket_list",
            "result": {
                "total": len(tickets),
                "items": tickets,
            },
        }
    if intent == "ticket_create":
        draft = generate_ticket_draft(message)

        return {
            "intent": "ticket_create",
            "requires_confirmation": True,
            "message": "已生成工单草稿，请确认后再创建。",
            "draft": draft,
        }
    return {
        "intent": "unknown",
        "message": "暂时无法判断你的需求。你可以询问知识库内容，或让我查看工单。",
    }

@app.post("/agent/tickets/confirm")
def confirm_ticket_creation(
    request: TicketConfirmRequest,
    db: Session = Depends(get_db),
):
    title = request.title.strip()
    content = request.content.strip()

    if not title:
        raise HTTPException(
            status_code=400,
            detail="工单标题不能为空",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="工单内容不能为空",
        )

    ticket = Ticket(
        title=title,
        content=content,
        priority=request.priority,
    )

    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return {
        "message": "工单已确认创建",
        "ticket": ticket,
    }