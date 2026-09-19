"""Search_ArXiv 解析测试。不访问真实 arXiv。"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

if str(Path(__file__).parent.parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.tools.openreview import parse_arxiv_atom, search_arxiv_func

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1505.04597v1</id>
    <title>U-Net: Convolutional Networks
for Biomedical Image Segmentation</title>
    <summary>A network for biomedical segmentation.</summary>
    <published>2015-05-18T00:00:00Z</published>
    <author><name>Olaf Ronneberger</name></author>
    <author><name>Philipp Fischer</name></author>
    <link href="http://arxiv.org/pdf/1505.04597v1" title="pdf" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


def test_parse_arxiv_atom_extracts_required_fields():
    papers = parse_arxiv_atom(_FEED)
    assert len(papers) == 1
    paper = papers[0]
    assert paper["title"] == "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    assert paper["authors"] == ["Olaf Ronneberger", "Philipp Fischer"]
    assert paper["abstract"] == "A network for biomedical segmentation."
    assert paper["arxiv_id"] == "1505.04597"
    assert paper["published"] == "2015-05-18T00:00:00Z"
    assert paper["pdf_url"] == "http://arxiv.org/pdf/1505.04597v1"


def test_search_arxiv_func_uses_mocked_http_response():
    response = Mock()
    response.text = _FEED
    response.raise_for_status = Mock()
    with patch("agents.tools.openreview.httpx.get", return_value=response) as get:
        raw = search_arxiv_func("medical image segmentation", max_results=3)

    params = get.call_args.kwargs["params"]
    assert params["max_results"] == 3
    assert "medical image segmentation" in params["search_query"]
    data = json.loads(raw)
    assert data["total_papers"] == 1
    assert data["papers"][0]["arxiv_id"] == "1505.04597"
