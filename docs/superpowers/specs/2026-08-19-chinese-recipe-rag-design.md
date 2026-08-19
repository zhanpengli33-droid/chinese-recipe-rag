# 中式食谱 RAG 智能问答系统设计规格

## 目标

构建一个与简历描述一致、无需 API Key 即可运行的中式食谱 RAG 项目。项目以 README 和关键代码展示工程能力，不实现前端、数据库或复杂部署。

## 对外命名

- 本地目录与 GitHub 仓库名：`chinese-recipe-rag`
- README 标题：`Chinese Recipe RAG System`
- README 副标题：`基于混合检索与 RRF 融合重排的中式食谱智能问答系统`
- Python 发行包名：`chinese-recipe-rag`
- Python 导入包名：`recipe_rag`
- GitHub 简介：`An offline Chinese recipe RAG system with hybrid retrieval, RRF reranking, query routing, evaluation, and streaming FastAPI responses.`

## 范围与边界

### 包含

- 离线语义检索、BM25 关键词检索和 RRF 融合重排。
- 菜品推荐、食材查询和分步做法三类查询路由。
- 简单口语改写和基于检索证据的模板回答。
- FastAPI 检索接口和 SSE 流式问答接口。
- Hit@5 指标计算与基线/优化链路对比。
- 使用固定随机种子生成 323 条本地模拟食谱和 120 条评测查询。
- 预留可选 BGE Embedding 与 FAISS 适配器，默认测试和运行不下载模型。

### 不包含

- 前端页面、用户系统、数据库、Docker 多服务编排和线上部署。
- 外部大模型 API 和必需的第三方密钥。
- “生产级”、“已上线”或真实用户效果等无法由仓库支撑的描述。

## 组件设计

### `retriever.py`

集中实现 BM25、确定性离线向量检索和 RRF 融合。中文分词使用字级 unigram/bigram；默认语义后端将特征稳定哈希到 256 维向量并使用余弦相似度排序。对外暴露统一的 `HybridRetriever.search(query, top_k)` 接口。可选的 BGE/FAISS 后端与默认后端使用相同接口，放在 `full` 可选依赖组中。

### `pipeline.py`

完成查询类型识别、口语改写、调用混合检索，并根据召回食谱生成包含菜名、食材、步骤和来源标识的模板回答。

### `api.py`

提供 `GET /health`、`POST /search` 和 `POST /chat/stream` 三个接口。SSE 按 `route`、`retrieval`、`answer` 顺序输出阶段事件。

### `evaluation.py`

根据预期命中的食谱 ID 计算 Hit@5，输出单路基线和融合链路的对比结果。固定评测快照使用基线 82/120 命中和融合链路 106/120 命中，计算结果为 68.3%、88.3% 和相对提升 29.3%。README 按简历口径四舍五入为约 68%、88% 和 29%。

### `scripts/build_demo_data.py`

根据少量菜系、主食材、烹饪方式和步骤模板可重复生成 323 条食谱、120 条评测查询以及对应排名快照。产物作为包内数据提交到 `src/recipe_rag/data/`，并明确标记为模拟数据，不声称来自真实业务库。

## 数据流

`UserQuery -> QueryRouter/Rewriter -> BM25 + Dense Retrieval -> RRF -> Template Generator -> JSON/SSE Response`

## 错误处理

- 空查询和超长查询由 API 输入模型拒绝并返回 `400`。
- 检索结果为空时返回低置信度降级回答，不伪造食谱。
- BGE/FAISS 可选依赖缺失时回退到默认离线后端。
- SSE 执行失败时输出结构化 `error` 事件并正常结束流。
- 数据文件缺失或格式错误时快速失败，错误信息包含文件路径和字段原因。

## 测试设计

关键测试控制在约 10 项，覆盖：

- 模拟数据严格生成 323 条食谱和 120 条评测查询。
- 查询路由区分推荐、食材和步骤问答。
- BM25、离线向量检索和 RRF 返回稳定排序。
- Hit@5 计算口径正确。
- API 返回结构化检索结果，SSE 输出预期事件顺序。
- 空查询、空召回和可选依赖缺失时的降级逻辑。

## README 包装

README 重点展示问题背景、架构图、双路检索与 RRF、查询优化、FastAPI/SSE、评测口径和本地运行方式。必须明确食谱、查询和指标均为本地模拟/自建数据，不暗示真实线上用户效果。

## 验收标准

- 无 API Key 可安装并运行核心流程。
- 关键测试全部通过，工作区无未提交改动。
- README 项目名、技术链路、数据量和评测口径与简历一致。
- GitHub 仓库保持公开，地址为 `https://github.com/zhanpengli33-droid/chinese-recipe-rag`。
- GitHub Topics 使用 `rag`、`hybrid-search`、`rrf`、`fastapi`、`sse`、`python`。
