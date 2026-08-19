from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import requests
import streamlit as st


API_BASE = os.getenv("RESEARCHPILOT_API_BASE", "http://localhost:8000/api")
API_ORIGIN = API_BASE[:-4] if API_BASE.endswith("/api") else API_BASE

st.set_page_config(page_title="ResearchPilot", page_icon="RP", layout="wide")
st.title("ResearchPilot")
st.caption("面向个人科研资料的 RAG 知识助手（自动分库 / 查询扩展 / 低置信二次检索）")


def _refresh_kb_list() -> None:
    st.session_state["kb_list"] = requests.get(f"{API_BASE}/knowledge-bases", timeout=30).json()
    st.session_state["kb_stats"] = requests.get(
        f"{API_BASE}/knowledge-bases/stats", timeout=30
    ).json()


def _refresh_docs() -> None:
    st.session_state["docs"] = requests.get(f"{API_BASE}/documents", timeout=30).json()


def _media_url(citation: dict[str, Any]) -> str | None:
    image_url = citation.get("image_url")
    if image_url:
        return f"{API_ORIGIN}{image_url}" if str(image_url).startswith("/") else str(image_url)
    image_path = citation.get("image_path")
    if image_path:
        return f"{API_BASE}/media?path={quote(str(image_path), safe='')}"
    return None


with st.sidebar:
    st.header("知识库")
    if st.button("刷新知识库列表"):
        try:
            _refresh_kb_list()
        except requests.RequestException as exc:
            st.error(str(exc))
    for item in st.session_state.get("kb_list", []):
        kb_id = str(item.get("id"))
        cols = st.columns([4, 1])
        with cols[0]:
            st.write(f"**{item.get('name')}** (`{kb_id}`)")
            if item.get("description"):
                st.caption(str(item["description"])[:120])
        with cols[1]:
            if st.button("删库", key=f"del_kb_{kb_id}"):
                try:
                    resp = requests.delete(
                        f"{API_BASE}/knowledge-bases/{kb_id}", timeout=60
                    )
                    if resp.ok:
                        st.success(f"已删除库 {kb_id}")
                        _refresh_kb_list()
                        _refresh_docs()
                        st.rerun()
                    else:
                        st.error(resp.text)
                except requests.RequestException as exc:
                    st.error(str(exc))
    if "kb_stats" in st.session_state:
        with st.expander("统计"):
            st.json(st.session_state["kb_stats"])

    st.divider()
    st.header("CLIP 重索引（B3）")
    st.caption("用仍存在的源文件按当前 embedding（CLIP）重建向量；找不到源文件的会跳过并列出。")
    if st.button("重索引现有文档", type="primary"):
        try:
            with st.spinner("正在按源文件重新索引（可能较久）..."):
                resp = requests.post(
                    f"{API_BASE}/documents/reindex-existing",
                    data={"chunk_strategy": "heading"},
                    timeout=3600,
                )
            st.session_state["last_reindex"] = {
                "ok": resp.ok,
                "payload": resp.json() if resp.ok else {"detail": resp.text},
            }
            if resp.ok:
                _refresh_kb_list()
                _refresh_docs()
        except requests.RequestException as exc:
            st.session_state["last_reindex"] = {
                "ok": False,
                "payload": {"detail": str(exc)},
            }
    last_reindex = st.session_state.get("last_reindex")
    if last_reindex:
        if last_reindex.get("ok"):
            payload = last_reindex.get("payload") or {}
            st.success(
                f"完成：重索引 {payload.get('reindexed_count', 0)}，"
                f"跳过 {payload.get('skipped_count', 0)}；"
                f"image_chunks={((payload.get('stats') or {}).get('image_chunks'))}"
            )
            with st.expander("重索引报告"):
                st.json(payload)
        else:
            st.error((last_reindex.get("payload") or {}).get("detail", "重索引失败"))

    st.divider()
    st.header("文档导入")
    st.caption("上传后由文本 LLM 自动分配到已有知识库，或创建新库。")
    uploaded = st.file_uploader(
        "上传文件 / 图片",
        type=[
            "pdf",
            "md",
            "markdown",
            "txt",
            "log",
            "csv",
            "xlsx",
            "xls",
            "docx",
            "pptx",
            "ppt",
            "html",
            "htm",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "bmp",
            "tif",
            "tiff",
        ],
    )
    title = st.text_input("标题，可选")
    strategy = st.selectbox("Chunk 策略", ["heading", "recursive", "fixed"])
    force_reindex = st.toggle("强制重新索引", value=False)
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=4000, value=900, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=120, step=20)
    if st.button("索引文档", disabled=uploaded is None):
        files = {"file": (uploaded.name, uploaded.getvalue())} if uploaded else None
        data = {
            "title": title or "",
            "chunk_strategy": strategy,
            "force_reindex": str(force_reindex).lower(),
            "chunk_size": str(chunk_size),
            "chunk_overlap": str(chunk_overlap),
        }
        try:
            with st.spinner("正在解析、自动分库并索引（含 VLM，大文件可能较慢）..."):
                response = requests.post(
                    f"{API_BASE}/documents/upload",
                    files=files,
                    data=data,
                    timeout=900,
                )
            try:
                payload = response.json()
            except ValueError:
                payload = {"detail": response.text}
            st.session_state["last_upload"] = {
                "ok": response.ok,
                "status_code": response.status_code,
                "payload": payload,
            }
        except requests.RequestException as exc:
            st.session_state["last_upload"] = {
                "ok": False,
                "status_code": None,
                "payload": {
                    "detail": str(exc),
                    "hint": "前端等待超时或连接失败。后端可能仍在处理。",
                },
            }

    last_upload = st.session_state.get("last_upload")
    if last_upload:
        st.subheader("解析路由")
        if not last_upload.get("ok"):
            payload = last_upload.get("payload") or {}
            st.error(
                f"上传失败 (HTTP {last_upload.get('status_code')}): "
                f"{payload.get('detail', payload)}"
            )
            if payload.get("hint"):
                st.warning(payload["hint"])
        route = (last_upload.get("payload") or {}).get("route") or {}
        st.json(route)
        assignment = (last_upload.get("payload") or {}).get("knowledge_base_assignment")
        if assignment:
            st.subheader("知识库分配")
            st.json(assignment)
        st.subheader("索引结果")
        st.json(last_upload.get("payload"))

    st.divider()
    st.header("全部文档")
    if st.button("刷新文档列表"):
        try:
            _refresh_docs()
        except requests.RequestException as exc:
            st.error(str(exc))
    for doc in st.session_state.get("docs", []):
        meta = doc.get("metadata") or {}
        doc_id = str(doc["id"])
        cols = st.columns([4, 1])
        with cols[0]:
            st.write(
                f"- {doc['title']} | kb=`{meta.get('knowledge_base_id')}` "
                f"({doc['chunk_count']} chunks)"
            )
        with cols[1]:
            if st.button("删除", key=f"del_doc_{doc_id}"):
                try:
                    resp = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=60)
                    if resp.ok:
                        st.success("文档已删除")
                        _refresh_docs()
                        st.rerun()
                    else:
                        st.error(resp.text)
                except requests.RequestException as exc:
                    st.error(str(exc))

question = st.text_area("输入科研问题", placeholder="例如：这篇论文的核心方法和实验结论是什么？")
top_k = st.slider("Top K", min_value=1, max_value=12, value=6)
use_rerank = st.toggle("启用 rerank", value=True)

st.subheader("图搜（CLIP 多模态）")
st.caption(
    "纯图：跳过路由与文字改写，只走 CLIP 图向量；生成时把查询图/召回图交给 VLM。"
    "有文字时先按文字路由，不确定则自动降级全库。命中图片会在下方画廊展示。"
)
query_image = st.file_uploader(
    "上传查询图片（可选，与上方文字可同时使用）",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    key="query_image",
)

col_ask, col_img = st.columns(2)
with col_ask:
    ask_clicked = st.button("文字提问", disabled=not question.strip())
with col_img:
    img_ask_clicked = st.button("图片检索/问答", disabled=query_image is None)

if ask_clicked:
    st.session_state["pending_query_image"] = None
    payload = {
        "question": question,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "auto_route": True,
    }
    with st.spinner("正在路由、查询扩展、检索（必要时二次检索）并用 LLM 生成回答..."):
        response = requests.post(f"{API_BASE}/ask", json=payload, timeout=240)
    data = cast(dict[str, Any], response.json())
    st.session_state["last_ask"] = data
    st.session_state["pending_question"] = question

if img_ask_clicked and query_image is not None:
    image_bytes = query_image.getvalue()
    st.session_state["pending_query_image"] = {
        "name": query_image.name,
        "bytes": image_bytes,
    }
    files = {"file": (query_image.name, image_bytes)}
    data_form = {
        "question": question or "",
        "top_k": str(top_k),
        "use_rerank": str(use_rerank).lower(),
        "auto_route": "true",
    }
    with st.spinner("正在用 CLIP 做图搜（图搜图/图搜文）并生成回答..."):
        response = requests.post(
            f"{API_BASE}/ask/image",
            files=files,
            data=data_form,
            timeout=300,
        )
    if response.ok:
        st.session_state["last_ask"] = cast(dict[str, Any], response.json())
        st.session_state["pending_question"] = question or "（图片查询）"
    else:
        st.error(response.text)

last_ask = st.session_state.get("last_ask")
if last_ask:
    routing = last_ask.get("routing") or {}
    st.subheader("查询路由")
    st.json(routing)

    if last_ask.get("query_plan"):
        st.subheader("查询扩展计划")
        st.json(last_ask.get("query_plan"))
    if last_ask.get("retrieval_retry"):
        st.subheader("二次检索")
        st.json(last_ask.get("retrieval_retry"))

    if last_ask.get("need_selection"):
        st.warning(last_ask.get("message") or "请选择知识库")
        candidates = routing.get("candidates") or []
        options = {
            f"{item.get('name')} ({item.get('id')})": item.get("id") for item in candidates
        }
        chosen_labels = st.multiselect(
            "选择 1～3 个知识库后继续",
            options=list(options.keys()),
            max_selections=3,
        )
        if st.button("用所选知识库继续提问", disabled=not chosen_labels):
            selected_ids = [options[label] for label in chosen_labels if label in options]
            pending_image = st.session_state.get("pending_query_image")
            with st.spinner("正在检索并用 LLM 生成回答..."):
                if pending_image:
                    response = requests.post(
                        f"{API_BASE}/ask/image",
                        files={"file": (pending_image["name"], pending_image["bytes"])},
                        data={
                            "question": st.session_state.get("pending_question") or question or "",
                            "top_k": str(top_k),
                            "use_rerank": str(use_rerank).lower(),
                            "auto_route": "false",
                            "selected_knowledge_base_ids": ",".join(str(i) for i in selected_ids),
                        },
                        timeout=300,
                    )
                else:
                    payload = {
                        "question": st.session_state.get("pending_question") or question,
                        "top_k": top_k,
                        "use_rerank": use_rerank,
                        "auto_route": False,
                        "selected_knowledge_base_ids": selected_ids,
                    }
                    response = requests.post(f"{API_BASE}/ask", json=payload, timeout=240)
            st.session_state["last_ask"] = cast(dict[str, Any], response.json())
            st.rerun()
    else:
        st.subheader("回答")
        st.write(last_ask.get("answer"))
        cols = st.columns(2)
        cols[0].metric("置信度", last_ask.get("confidence", "unknown"))
        cols[1].metric("生成器", str(last_ask.get("generator", "unknown")))

        citations = [cast(dict[str, Any], c) for c in last_ask.get("citations", [])]
        image_citations = [c for c in citations if _media_url(c)]
        if image_citations:
            st.subheader("检索到的图片")
            gallery_cols = st.columns(min(3, len(image_citations)))
            for idx, citation_data in enumerate(image_citations):
                with gallery_cols[idx % len(gallery_cols)]:
                    url = _media_url(citation_data)
                    if url:
                        st.image(
                            url,
                            caption=(
                                f"{citation_data.get('document_title')} | "
                                f"score={citation_data.get('score')}"
                            ),
                        )

        st.subheader("引用来源")
        for citation_data in citations:
            label = f"{citation_data.get('document_title')} | score={citation_data.get('score')}"
            if citation_data.get("modality") == "image" or citation_data.get("image_path"):
                label = f"[图] {label}"
            with st.expander(label):
                st.json(citation_data)
                url = _media_url(citation_data)
                if url:
                    st.image(url, caption=str(citation_data.get("image_path") or url))
                else:
                    image_path = citation_data.get("image_path")
                    if image_path and Path(str(image_path)).exists():
                        st.image(str(image_path), caption=str(image_path))

        st.subheader("检索 Trace")
        st.json(last_ask.get("trace", []))
