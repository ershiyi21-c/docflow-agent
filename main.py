from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="DocFlow Agent",
    description="企业知识库与工单协同 Agent",
    version="0.1.0",
)

# 临时保存工单的列表。
# 程序重启后，这里面的数据会消失。
tickets = []


class TicketCreate(BaseModel):
    title: str
    content: str
    priority: str = "normal"


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "DocFlow Agent API is running",
    }


@app.post("/tickets")
def create_ticket(ticket: TicketCreate):
    ticket_data = ticket.model_dump()

    new_ticket = {
        "id": str(uuid4()),
        "status": "open",
        **ticket_data,
    }

    tickets.append(new_ticket)

    return {
        "message": "工单创建成功",
        "ticket": new_ticket,
    }


@app.get("/tickets")
def list_tickets():
    return {
        "total": len(tickets),
        "items": tickets,
    }