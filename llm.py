# 本模块负责配置 DeepSeek 的 OpenAI 兼容客户端，并封装模型调用。
import os
import json
from dotenv import load_dotenv
from openai import OpenAI


# 将 .env 中的环境变量加载到当前进程。
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

# 在应用启动阶段尽早暴露密钥配置问题。
if not api_key:
    raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# 根据检索到的知识库上下文生成回答，并约束模型不得脱离资料作答。
def generate_knowledge_answer(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业知识库助手。"
                    "只能依据用户提供的资料回答。"
                    "资料不足时，明确说明资料中没有足够信息。"
                    "回答要简洁、准确。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"知识库资料：\n{context}"
                ),
            },
        ],
    )

    # 兼容模型响应内容为空的情况，保证调用方始终获得字符串。
    return response.choices[0].message.content or "模型没有返回内容。"

# 将自然语言问题压缩为适合字面检索的单个中文关键词。
def generate_search_keyword(question: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "你负责为企业知识库提取检索关键词。"
                    "只返回一个最关键的中文词或短语，最多 6 个字。"
                    "不要解释，不要标点，不要引号。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}",
            },
        ],
    )

    # 清理模型可能附带的首尾空白和常见标点。
    keyword = (response.choices[0].message.content or "").strip()
    keyword = keyword.strip("。！？,.，；;：:\"'“”")

    if not keyword:
        raise RuntimeError("模型没有返回检索关键词。")

    return keyword

def classify_agent_intent(message: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业知识库与工单协同 Agent 的意图分类器。"
                    "请判断用户消息属于 knowledge、ticket_list、ticket_create 或 unknown。"
                    "knowledge 表示制度、流程、文档知识相关问题。"
                    "ticket_list 表示查看、查询、列出工单。"
                    "ticket_create 表示用户希望创建、提交、登记或反馈一个工单问题。"
                    "unknown 表示无法判断或不属于上述类型。"
                    "只返回 JSON，不要解释。"
                    '格式必须是：{"intent":"knowledge"}'
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        response_format={
            "type": "json_object",
        },
    )


    content = response.choices[0].message.content or "{}"

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
        }

    intent = result.get("intent")

    if intent not in {
        "knowledge",
        "ticket_list",
        "ticket_create",
        "unknown",
    }:
        intent = "unknown"

    return {
        "intent": intent,
    }

def generate_ticket_draft(message: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "你负责从用户消息中提取工单草稿。"
                    "只返回 JSON，不要解释。"
                    '格式必须是：{"title":"工单标题","content":"问题详情","priority":"normal"}'
                    "priority 只能是 high、normal、low。"
                    "用户明确说高优先级、紧急、严重时使用 high。"
                    "没有明确优先级时使用 normal。"
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        response_format={
            "type": "json_object",
        },
    )

    content = response.choices[0].message.content or "{}"

    try:
        draft = json.loads(content)
    except json.JSONDecodeError:
        draft = {}

    title = str(draft.get("title", "")).strip()
    ticket_content = str(draft.get("content", "")).strip()
    priority = str(draft.get("priority", "normal")).strip().lower()

    if not title:
        title = "待确认工单"

    if not ticket_content:
        ticket_content = message.strip()

    if priority not in {"high", "normal", "low"}:
        priority = "normal"

    return {
        "title": title,
        "content": ticket_content,
        "priority": priority,
    }
