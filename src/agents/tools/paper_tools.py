"""
论文工具模块
包含论文搜索、下载等功能
"""
import json
import os
import re
import time
import httpx
import logging
from langchain_core.tools import BaseTool, tool

from agents.tools.utils import DOWNLOAD_PAPERS_DIR


def openreview_search_func(
    venue: str = "",
    domain: str = "",
    invitation: str = "",
    keyword: str = "",
    max_papers: int = 100,
    limit_per_page: int = 25,
) -> str:
    """Searches OpenReview API for academic papers.
    
    Useful for when you need to find papers from conferences like ICML, NeurIPS, ICLR, etc.
    This tool fetches papers from OpenReview API with pagination support.
    You can search by venue or by keyword. If keyword is provided, it will search across 
    multiple conferences and filter results by title/abstract containing the keyword.
    
    Args:
        venue (str): The venue filter (e.g., "ICML 2025 oral", "NeurIPS 2024"). 
                     If empty, will search across multiple conferences.
        domain (str): The conference domain (e.g., "ICML.cc/2025/Conference"). 
                     If empty and venue is provided, will try to infer from venue.
        invitation (str): The submission invitation path. If empty, will use default.
        keyword (str): Keyword to search for in paper titles/abstracts. 
                       If provided, will filter results to papers containing this keyword.
        max_papers (int): Maximum number of papers to fetch. Default: 100
        limit_per_page (int): Number of papers per page. Default: 25
    
    Returns:
        str: JSON string containing the list of papers with their details.
    """
    logger = logging.getLogger(__name__)
    base_url = "https://api2.openreview.net/notes"
    all_notes = []
    
    # 如果没有指定 venue，尝试搜索多个常见会议
    venues_to_try = []
    if venue:
        venues_to_try = [(venue, domain or "", invitation or "")]
    else:
        # 默认搜索多个会议
        venues_to_try = [
            ("NeurIPS 2024", "NeurIPS.cc/2024/Conference", "NeurIPS.cc/2024/Conference/-/Submission"),
            ("ICLR 2024", "ICLR.cc/2024/Conference", "ICLR.cc/2024/Conference/-/Submission"),
            ("ICML 2024", "ICML.cc/2024/Conference", "ICML.cc/2024/Conference/-/Submission"),
            ("NeurIPS 2023", "NeurIPS.cc/2023/Conference", "NeurIPS.cc/2023/Conference/-/Submission"),
        ]
    
    try:
        for venue_name, venue_domain, venue_invitation in venues_to_try:
            if not venue_domain:
                continue
                
            for offset in range(0, max_papers, limit_per_page):
                params = {
                    "content.venue": venue_name,
                    "details": "replyCount,presentation,writable",
                    "domain": venue_domain,
                    "invitation": venue_invitation,
                    "limit": limit_per_page,
                    "offset": offset,
                }
                
                try:
                    response = httpx.get(base_url, params=params, timeout=30.0)
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data.get("notes"):
                        break
                    
                    # 如果有关键词，过滤结果
                    if keyword:
                        filtered_notes = []
                        keyword_lower = keyword.lower()
                        for note in data["notes"]:
                            title = str(note.get("content", {}).get("title", "")).lower()
                            abstract = str(note.get("content", {}).get("abstract", "")).lower()
                            if keyword_lower in title or keyword_lower in abstract:
                                filtered_notes.append(note)
                        all_notes.extend(filtered_notes)
                    else:
                        all_notes.extend(data["notes"])
                    
                    # 如果已经找到足够的论文，停止搜索
                    if len(all_notes) >= max_papers:
                        break
                    
                    # Rate limiting
                    time.sleep(0.5)
                except httpx.HTTPError as e:
                    # 如果某个会议搜索失败，继续尝试下一个
                    logger.warning(f"Failed to search {venue_name}: {e}")
                    break
            
            # 如果已经找到足够的论文，停止搜索其他会议
            if len(all_notes) >= max_papers:
                break
        
        # 限制返回的论文数量
        all_notes = all_notes[:max_papers]
        
        # Format the results
        result = {
            "total_papers": len(all_notes),
            "papers": [
                {
                    "id": note.get("id"),
                    "title": note.get("content", {}).get("title", "N/A"),
                    "abstract": note.get("content", {}).get("abstract", "N/A"),
                    "authors": note.get("content", {}).get("authors", []),
                    "venue": note.get("content", {}).get("venue", "N/A"),
                    "keywords": note.get("content", {}).get("keywords", []),
                    "pdf_url": f"https://openreview.net/pdf?id={note.get('id')}" if note.get("id") else None,
                    "openreview_url": f"https://openreview.net/forum?id={note.get('id')}" if note.get("id") else None,
                }
                for note in all_notes
            ],
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error processing OpenReview data: {e}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "total_papers": 0,
            "papers": [],
            "error": error_msg
        }, indent=2, ensure_ascii=False)


openreview_search: BaseTool = tool(openreview_search_func)
openreview_search.name = "OpenReview_Search"


def download_paper_func(
    paper_id: str,
    save_path: str = "",
    download_dir: str = ""
) -> str:
    """下载 OpenReview 论文的 PDF 文件。
    
    从 OpenReview 下载指定论文的 PDF 文件并保存到本地。
    默认保存到统一的数据目录下的 downloads/papers 文件夹。
    
    Args:
        paper_id: OpenReview 论文 ID（例如从 openreview_search 获取的 id）
        save_path: 保存文件的完整路径（可选，如果不提供则自动生成）
        download_dir: 下载目录（可选，如果不提供则使用统一目录 ./data/downloads/papers）
    
    Returns:
        str: JSON 字符串，包含下载结果信息
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 确保下载目录存在（使用统一路径）
        if download_dir == "":
            # 优先使用环境变量，如果没有则使用统一目录
            # 注意：如果环境变量指向旧路径，也使用统一目录
            env_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "")
            if env_dir and env_dir != "./downloads" and env_dir != "./downloads/papers":
                # 如果环境变量设置了且不是旧路径，使用环境变量
                download_dir = env_dir
            else:
                # 否则使用统一目录
                download_dir = DOWNLOAD_PAPERS_DIR
        os.makedirs(download_dir, exist_ok=True)
        
        # 构建 PDF URL
        pdf_url = f"https://openreview.net/pdf?id={paper_id}"
        
        # 如果没有指定保存路径，自动生成
        if not save_path:
            # 先获取论文标题作为文件名
            try:
                # 获取论文详细信息以获取标题
                detail_url = f"https://api2.openreview.net/notes?id={paper_id}"
                detail_response = httpx.get(detail_url, timeout=30.0)
                detail_response.raise_for_status()
                detail_data = detail_response.json()
                
                if detail_data.get("notes"):
                    title = detail_data["notes"][0].get("content", {}).get("title", "paper")
                    # 清理文件名（移除非法字符）
                    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:100]  # 限制长度
                    filename = f"{safe_title}_{paper_id[:8]}.pdf"
                else:
                    filename = f"paper_{paper_id[:8]}.pdf"
            except Exception:
                # 如果获取标题失败，使用默认文件名
                filename = f"paper_{paper_id[:8]}.pdf"
            
            save_path = os.path.join(download_dir, filename)
        
        # 下载 PDF
        logger.info(f"正在下载论文 PDF: {pdf_url}")
        response = httpx.get(pdf_url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        
        # 检查是否是 PDF 内容
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
            # 可能返回的是 HTML 页面（论文未公开），尝试其他方式
            raise ValueError(f"无法下载 PDF，可能论文未公开或需要登录。Content-Type: {content_type}")
        
        # 保存文件
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        file_size = len(response.content) / 1024  # KB
        
        return json.dumps({
            "success": True,
            "paper_id": paper_id,
            "file_path": save_path,
            "file_size_kb": round(file_size, 2),
            "message": f"论文 PDF 已成功下载到: {save_path}"
        }, indent=2, ensure_ascii=False)
        
    except httpx.HTTPError as e:
        error_msg = f"下载论文时 HTTP 错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "paper_id": paper_id,
            "error": error_msg
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        error_msg = f"下载论文时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "paper_id": paper_id,
            "error": error_msg
        }, indent=2, ensure_ascii=False)


download_paper: BaseTool = tool(download_paper_func)
download_paper.name = "Download_Paper"


def download_paper_from_arxiv_func(
    arxiv_id: str = "",
    arxiv_url: str = "",
    save_path: str = "",
    download_dir: str = ""
) -> str:
    """从 arXiv 下载论文的 PDF 文件。
    
    支持通过 arXiv ID 或 URL 下载论文。
    例如：arxiv_id="1706.03762" 或 arxiv_url="https://arxiv.org/abs/1706.03762"
    
    Args:
        arxiv_id: arXiv 论文 ID（例如：1706.03762，不需要 "arXiv:" 前缀）
        arxiv_url: arXiv 论文 URL（例如：https://arxiv.org/abs/1706.03762）
        save_path: 保存文件的完整路径（可选，如果不提供则自动生成）
        download_dir: 下载目录（可选，如果不提供则使用统一目录 ./data/downloads/papers）
    
    Returns:
        str: JSON 字符串，包含下载结果信息
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 从 URL 或 ID 提取 arXiv ID
        if arxiv_url:
            # 从 URL 提取 ID
            match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', arxiv_url)
            if match:
                arxiv_id = match.group(1)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"无法从 URL 中提取 arXiv ID: {arxiv_url}"
                }, indent=2, ensure_ascii=False)
        
        if not arxiv_id:
            return json.dumps({
                "success": False,
                "error": "必须提供 arxiv_id 或 arxiv_url"
            }, indent=2, ensure_ascii=False)
        
        # 清理 arXiv ID（移除 "arXiv:" 前缀等）
        arxiv_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "").strip()
        
        # 确保下载目录存在
        if download_dir == "":
            env_dir = os.getenv("OPENREVIEW_DOWNLOAD_DIR", "")
            if env_dir and env_dir != "./downloads" and env_dir != "./downloads/papers":
                download_dir = env_dir
            else:
                download_dir = DOWNLOAD_PAPERS_DIR
        os.makedirs(download_dir, exist_ok=True)
        
        # 构建 arXiv PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # 如果没有指定保存路径，自动生成
        if not save_path:
            # 尝试获取论文标题
            try:
                abs_url = f"https://arxiv.org/abs/{arxiv_id}"
                abs_response = httpx.get(abs_url, timeout=30.0)
                abs_response.raise_for_status()
                
                # 从 HTML 中提取标题
                title_match = re.search(r'<title>([^<]+)</title>', abs_response.text)
                if title_match:
                    title = title_match.group(1).replace("arXiv:", "").strip()
                    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:100]
                    filename = f"{safe_title}_{arxiv_id}.pdf"
                else:
                    filename = f"arxiv_{arxiv_id}.pdf"
            except Exception:
                filename = f"arxiv_{arxiv_id}.pdf"
            
            save_path = os.path.join(download_dir, filename)
        
        # 下载 PDF
        logger.info(f"正在从 arXiv 下载论文 PDF: {pdf_url}")
        response = httpx.get(pdf_url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        
        # 检查是否是 PDF 内容
        if not response.content.startswith(b"%PDF"):
            return json.dumps({
                "success": False,
                "error": f"下载的内容不是 PDF 文件，可能 arXiv ID 不正确: {arxiv_id}"
            }, indent=2, ensure_ascii=False)
        
        # 保存文件
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        file_size = len(response.content) / 1024  # KB
        
        return json.dumps({
            "success": True,
            "arxiv_id": arxiv_id,
            "file_path": save_path,
            "file_size_kb": round(file_size, 2),
            "message": f"论文 PDF 已成功下载到: {save_path}"
        }, indent=2, ensure_ascii=False)
        
    except httpx.HTTPError as e:
        error_msg = f"从 arXiv 下载论文时 HTTP 错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "arxiv_id": arxiv_id if 'arxiv_id' in locals() else "",
            "error": error_msg
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        error_msg = f"从 arXiv 下载论文时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "arxiv_id": arxiv_id if 'arxiv_id' in locals() else "",
            "error": error_msg
        }, indent=2, ensure_ascii=False)


download_paper_from_arxiv: BaseTool = tool(download_paper_from_arxiv_func)
download_paper_from_arxiv.name = "Download_Paper_From_ArXiv"


def list_downloaded_papers_func() -> str:
    """
    列出已下载的论文文件。
    
    返回 ./data/downloads/papers/ 目录中的所有PDF文件列表。
    
    Returns:
        str: JSON字符串，包含文件列表
    """
    logger = logging.getLogger(__name__)
    
    try:
        papers = []
        if os.path.exists(DOWNLOAD_PAPERS_DIR):
            for file in os.listdir(DOWNLOAD_PAPERS_DIR):
                if file.endswith('.pdf'):
                    file_path = os.path.join(DOWNLOAD_PAPERS_DIR, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    papers.append({
                        "filename": file,
                        "path": file_path,
                        "size_kb": round(file_size, 2)
                    })
        
        return json.dumps({
            "success": True,
            "directory": DOWNLOAD_PAPERS_DIR,
            "total_files": len(papers),
            "papers": sorted(papers, key=lambda x: x["filename"])
        }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"列出文件时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "error": error_msg
        }, indent=2, ensure_ascii=False)


list_downloaded_papers: BaseTool = tool(list_downloaded_papers_func)
list_downloaded_papers.name = "List_Downloaded_Papers"

__all__ = [
    "openreview_search",
    "download_paper",
    "download_paper_from_arxiv",
    "list_downloaded_papers",
]

