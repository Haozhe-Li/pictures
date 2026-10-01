<div align="center">

# 📸 Pictures

### 面向摄影师的 AI 驱动多模态图库

[![在线演示](https://img.shields.io/badge/🌐_在线演示-haozheli.pictures-blue?style=for-the-badge)](https://www.haozheli.pictures/)

[English](README.md) | [简体中文](README_ZH.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1E1E1E?style=flat-square&logo=qdrant&logoColor=white)
![Cloudflare R2](https://img.shields.io/badge/Cloudflare_R2-F38020?style=flat-square&logo=cloudflare&logoColor=white)

**Pictures** 是一个生产就绪的多模态、多语言图库，结合了混合检索（稀疏 + 稠密）和智能缓存，可从自然语言或图像中提供快速、准确的搜索结果。

![主页演示](assets/pictures-home.gif)

[功能特性](#-功能特性) • [架构](#️-架构) • [技术栈](#️-技术栈) • [快速开始](#-快速开始) • [配置](#️-配置) • [部署](#-部署)

</div>

---

## ✨ 功能特性

### 🔍 自然语言搜索

由混合稀疏+稠密检索驱动，提供准确、上下文感知的结果。

![自然语言搜索](assets/pictures-search.gif)

### 🌐 多语言支持

支持任何语言搜索 — 无语言障碍，由 CLIP 嵌入驱动。

![多语言搜索](assets/pictures-multilingual.gif)

### 🖼️ 以图搜图

使用先进的视觉嵌入技术查找视觉相似的照片。

![以图搜图](assets/pictures-similar.gif)

### 🎲 智能推荐

通过智能随机推荐发现新照片。

![随机推荐](assets/pictures-random-search.gif)

### ⚡ 高性能

多层缓存架构最大限度地减少延迟并降低大规模成本。

---

## 🏗️ 架构

### 混合检索

Pictures 使用复杂的混合检索系统，结合了：

![混合检索流程](https://cdn.haozheli.com/pictures-flow-chart.webp)

1. **稀疏检索 (BM25)** — 基于关键词的元数据匹配
2. **稠密检索 (CLIP)** — 语义文本和图像嵌入
3. **向量存储 (Qdrant)** — 高效相似性搜索
4. **排名融合 (RRF)** — 倒数排名融合以获得最佳结果

### 多层缓存

智能缓存减少延迟和 API 成本：

![多层缓存](https://cdn.haozheli.com/pictures-cache-layer.webp)

- **第一层：** 前端缓存（Next.js + Upstash Redis）
- **第二层：** 后端服务缓存
- **第三层：** 嵌入缓存（查询向量）

---

## 🛠️ 技术栈

### 后端

- **框架：** FastAPI (Python 3.10+)
- **向量数据库：** Qdrant
- **嵌入：** CLIP (`jinaai/jina-clip-v1`，自托管 FastEmbed 服务), BM25 (FastEmbed)
- **检索：** BM25 + 稠密向量搜索
- **缓存：** Redis
- **存储：** Cloudflare R2

### 前端

- **框架：** Next.js 14 (App Router)
- **语言：** TypeScript
- **样式：** Tailwind CSS
- **UI 组件：** Radix UI + shadcn/ui
- **缓存：** Upstash Redis
- **部署：** Vercel

### 基础设施

- **CDN：** Cloudflare
- **容器：** Docker
- **监控：** 内置日志

---

## 📂 项目结构

```
gallery_RAG/
├── backend/                 # FastAPI 后端服务
│   ├── main.py             # 应用入口
│   ├── requirements.txt    # Python 依赖
│   ├── Dockerfile          # 容器配置
│   ├── core/               # 核心模块
│   │   ├── config.py       # 配置管理
│   │   ├── db.py           # 数据库连接
│   │   ├── embedding.py    # 嵌入生成
│   │   ├── storage.py      # R2 存储集成
│   │   └── utils.py        # 工具函数
│   └── models/             # 模型文件
├── frontend/               # Next.js 图库应用
│   ├── app/                # App 路由页面
│   │   ├── page.tsx        # 主页
│   │   └── api/            # API 路由
│   ├── components/         # React 组件
│   │   ├── gallery.tsx
│   │   ├── search-bar.tsx
│   │   └── ui/             # UI 基础组件
│   └── lib/                # 工具库
│       ├── api.ts          # API 客户端
│       └── types.ts        # TypeScript 类型
├── upload_GUI/             # 管理上传界面
│   ├── app/                # 上传应用页面
│   └── components/         # 上传组件
└── assets/                 # 文档资源
```

---

## 🚀 快速开始

### 前置要求

- **Python** 3.10 或更高版本
- **Node.js** 18 或更高版本
- **pnpm**（推荐）或 npm
- **Docker**（可选，用于容器化部署）
- **Redis** 实例（本地或云端）
- **Qdrant** 实例（本地或云端）

### 后端设置

1. **克隆仓库**

   ```bash
   git clone https://github.com/yourusername/gallery_RAG.git
   cd gallery_RAG/backend
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境变量**

   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入你的 API 密钥和配置
   ```

4. **运行后端**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   或使用 Docker：

   ```bash
   docker build -t pictures-backend .
   docker run -p 8000:8000 --env-file .env pictures-backend
   ```

### 前端设置

1. **进入前端目录**

   ```bash
   cd frontend
   ```

2. **安装依赖**

   ```bash
   pnpm install
   ```

3. **配置环境变量**

   ```bash
   cp .env.example .env.local
   # 编辑 .env.local 文件填入配置
   ```

4. **运行开发服务器**

   ```bash
   pnpm dev
   ```

   在浏览器中打开 [http://localhost:3000](http://localhost:3000)。

### 上传界面设置

1. **进入上传界面目录**

   ```bash
   cd upload_GUI
   ```

2. **安装依赖**

   ```bash
   pnpm install
   ```

3. **配置环境变量**

   ```bash
   cp .env.example .env.local
   # 设置 BACKEND_URL 为你的后端 URL
   ```

4. **运行上传界面**
   ```bash
   pnpm dev
   ```

---

## ⚙️ 配置

### 后端环境变量

在 `backend/` 目录中创建 `.env` 文件：

| 变量                | 描述                                         | 必需 |
| ------------------- | -------------------------------------------- | ---- |
| `CF_API_URL`        | Cloudflare R2 S3 端点 URL                    | ✅   |
| `CF_API_KEY_ID`     | R2 访问密钥 ID                               | ✅   |
| `CF_API_KEY_SECRET` | R2 访问密钥密文                              | ✅   |
| `QDRANT_URL`        | Qdrant 端点（默认：`http://localhost:6333`） | ✅   |
| `QDRANT_API_KEY`    | Qdrant API 密钥（如果使用云端）              | ⚠️   |
| `EMBEDDING_SERVICE_URL` | 嵌入服务的基础 URL（稠密向量）            | ✅   |
| `REDIS_HOST`        | Redis 主机地址                               | ✅   |
| `REDIS_PORT`        | Redis 端口（默认：`16666`）                  | ✅   |
| `REDIS_USERNAME`    | Redis 用户名（默认：`default`）              | ✅   |
| `REDIS_PASSWORD`    | Redis 密码                                   | ✅   |
| `ADMIN_USERNAME`    | `/ingest` 端点的基本认证用户名               | ✅   |
| `ADMIN_PASSWORD`    | `/ingest` 端点的基本认证密码                 | ✅   |

### 前端环境变量

在 `frontend/` 目录中创建 `.env.local` 文件：

| 变量                   | 描述                                          | 必需 |
| ---------------------- | --------------------------------------------- | ---- |
| `GALLERY_API_BASE_URL` | 后端基础 URL（例如：`http://localhost:8000`） | ✅   |
| `KV_REST_API_URL`      | Upstash Redis REST API URL                    | ✅   |
| `KV_REST_API_TOKEN`    | Upstash Redis REST API 令牌                   | ✅   |

### 上传界面环境变量

在 `upload_GUI/` 目录中创建 `.env.local` 文件：

| 变量          | 描述                   | 必需 |
| ------------- | ---------------------- | ---- |
| `BACKEND_URL` | 图像摄取的后端基础 URL | ✅   |

---

## 📚 API 文档

完整的 API 文档请参见 [backend/API.md](backend/API.md)。

### 主要端点

- `GET /gallery` — 获取所有图库图像
- `POST /search` — 按文本查询搜索图像
- `POST /similar` — 按图像 URL 查找相似图像
- `GET /random-query` — 获取随机搜索建议
- `POST /ingest` — 上传并处理新图像（仅管理员）

有关详细的请求/响应架构和示例，请参阅 [API 文档](backend/API.md)。

---

## 🚢 部署

### 生产架构

- **前端：** 部署在 Vercel，带有边缘缓存
- **后端：** 容器化部署（Docker）在任何云提供商
- **存储：** Cloudflare R2 with CDN
- **向量数据库：** Qdrant Cloud 或自托管
- **缓存：** Upstash Redis（前端）+ Redis（后端）

### 后端部署（Docker）

```bash
cd backend

# 构建 Docker 镜像
docker build -t pictures-backend .

# 运行容器
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name pictures-backend \
  pictures-backend
```

### 前端部署（Vercel）

1. **推送到 GitHub**

   ```bash
   git push origin main
   ```

2. **导入到 Vercel**
   - 访问 [vercel.com](https://vercel.com)
   - 导入你的仓库
   - 设置根目录为 `frontend/`
   - 从 `.env.local` 添加环境变量

3. **部署**
   - Vercel 会在每次推送到 `main` 时自动部署

### 上传界面部署

上传界面可以像前端一样部署，或仅为管理员使用在本地托管。确保正确配置 HTTP 基本认证凭据。

---

## 📄 许可证

该项目目前未授权。如果你计划开源此项目，请添加 LICENSE 文件（MIT、Apache 2.0 等）。

---

<div align="center">
**由 Haozhe Li 用 ❤️ 构建**

[⬆ 返回顶部](#-pictures)

</div>
