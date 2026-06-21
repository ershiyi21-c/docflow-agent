# DocFlow Agent

一个企业知识库与工单协同 Agent 项目。

## 当前功能

- FastAPI 后端服务
- `/health` 健康检查接口
- 自动生成 Swagger API 文档

## 技术栈

- Python 3.13
- FastAPI
- Uvicorn

## 本地运行

### 1. 创建并激活虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 启动项目

```powershell
uvicorn main:app --reload
```

启动后访问：

- 健康检查：`http://127.0.0.1:8000/health`
- API 文档：`http://127.0.0.1:8000/docs`

## 开发计划

- [ ] 工单 CRUD 接口
- [ ] SQLite 数据库
- [ ] 文档上传
- [ ] RAG 知识库问答
- [ ] Agent 工具调用
- [ ] 工单创建确认流程
- [ ] Docker 部署