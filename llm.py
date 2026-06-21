import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


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

    return response.choices[0].message.content or "模型没有返回内容。"