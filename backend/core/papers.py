"""
SMART ACADEMIC PAPER SEARCH — v2
Extracts scientific keywords from natural language queries,
searches multiple strategies, uses OpenAlex's built-in AI concepts.
No AI tokens used — pure algorithmic keyword extraction.

Save as: backend/core/papers.py (replaces the old version)
"""

from __future__ import annotations
import asyncio
import urllib.request
import urllib.parse
import json
import re

# Common stop words to strip from queries
STOP_WORDS = {
    'a','an','the','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','shall','can',
    'not','no','nor','but','and','or','if','then','else','when','where','why','how',
    'what','which','who','whom','this','that','these','those','i','me','my','we','our',
    'you','your','he','she','it','they','them','his','her','its','their','of','to',
    'in','for','on','with','at','by','from','as','into','about','between','through',
    'during','before','after','above','below','up','down','out','off','over','under',
    'again','further','very','just','some','all','any','each','few','more','most',
    'other','such','only','own','same','so','than','too','get','got','give','go',
    'make','say','tell','think','know','want','need','like','really','also','much',
    'many','because','since','while','although','though','please','explain','describe',
    'give','rid','something','new','come','take','find','look','see','way','thing',
}


def extract_keywords(query):
    """Extract scientific keywords from a natural language query."""
    # Lowercase and clean
    q = query.lower().strip()
    q = re.sub(r'[^\w\s\-]', ' ', q)

    # Split into words
    words = q.split()

    # Remove stop words
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    # Try to detect multi-word scientific terms
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in STOP_WORDS and words[i+1] not in STOP_WORDS:
            bigrams.append(words[i] + ' ' + words[i+1])

    # Combine: use the full cleaned query + extracted keywords
    return {
        'full': ' '.join(keywords) if keywords else query,
        'keywords': keywords[:6],
        'bigrams': bigrams[:3],
    }


async def _fetch_json(url, timeout=5):
    """Fetch JSON from a URL."""
    def do_fetch():
        req = urllib.request.Request(url, headers={
            "User-Agent": "Meridian/1.0 (mailto:meridian@research.app)",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    return await asyncio.to_thread(do_fetch)


async def search_semantic_scholar(query, keywords, limit=5):
    """Search Semantic Scholar with multiple strategies."""
    papers = []

    # Strategy 1: Search with extracted keywords
    try:
        keyword_query = ' '.join(keywords[:5]) if keywords else query
        encoded = urllib.parse.quote(keyword_query)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&fields=title,authors,year,citationCount,abstract,url,externalIds"
        data = await _fetch_json(url)

        for p in (data.get("data") or []):
            doi = (p.get("externalIds") or {}).get("DOI", "")
            papers.append({
                "title": p.get("title", ""),
                "authors": [a.get("name", "") for a in (p.get("authors") or [])[:4]],
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "abstract": (p.get("abstract") or "")[:300],
                "url": p.get("url") or (f"https://doi.org/{doi}" if doi else ""),
                "doi": doi,
                "source": "semantic_scholar",
            })
    except Exception as e:
        print(f"Semantic Scholar keyword search error: {e}")

    # Strategy 2: If few results, try with full original query
    if len(papers) < 2:
        try:
            encoded = urllib.parse.quote(query[:200])
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&fields=title,authors,year,citationCount,abstract,url,externalIds"
            data = await _fetch_json(url)

            existing_titles = {p["title"].lower()[:50] for p in papers}
            for p in (data.get("data") or []):
                title = p.get("title", "")
                if title.lower()[:50] not in existing_titles:
                    doi = (p.get("externalIds") or {}).get("DOI", "")
                    papers.append({
                        "title": title,
                        "authors": [a.get("name", "") for a in (p.get("authors") or [])[:4]],
                        "year": p.get("year"),
                        "citations": p.get("citationCount", 0),
                        "abstract": (p.get("abstract") or "")[:300],
                        "url": p.get("url") or (f"https://doi.org/{doi}" if doi else ""),
                        "doi": doi,
                        "source": "semantic_scholar",
                    })
        except Exception as e:
            print(f"Semantic Scholar full query error: {e}")

    return papers


async def search_openalex(query, keywords, limit=5):
    """Search OpenAlex with keyword strategy."""
    papers = []

    # Use keyword search
    keyword_query = ' '.join(keywords[:5]) if keywords else query
    try:
        encoded = urllib.parse.quote(keyword_query)
        url = f"https://api.openalex.org/works?search={encoded}&per_page={limit}&sort=relevance_score:desc&select=id,doi,title,authorships,publication_year,cited_by_count,abstract_inverted_index,primary_location"
        data = await _fetch_json(url)

        for w in (data.get("results") or []):
            abstract = ""
            inv = w.get("abstract_inverted_index")
            if inv:
                word_positions = []
                for word, positions in inv.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(wd for _, wd in word_positions)[:300]

            authors = []
            for a in (w.get("authorships") or [])[:4]:
                name = (a.get("author") or {}).get("display_name", "")
                if name:
                    authors.append(name)

            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            paper_url = ""
            loc = w.get("primary_location") or {}
            if loc.get("landing_page_url"):
                paper_url = loc["landing_page_url"]
            elif doi:
                paper_url = f"https://doi.org/{doi}"

            papers.append({
                "title": w.get("title", ""),
                "authors": authors,
                "year": w.get("publication_year"),
                "citations": w.get("cited_by_count", 0),
                "abstract": abstract,
                "url": paper_url,
                "doi": doi,
                "source": "openalex",
            })
    except Exception as e:
        print(f"OpenAlex search error: {e}")

    # Strategy 2: search by concept if few results
    if len(papers) < 2 and keywords:
        try:
            concept = urllib.parse.quote(keywords[0])
            url = f"https://api.openalex.org/works?filter=default.search:{concept}&per_page={limit}&sort=cited_by_count:desc&select=id,doi,title,authorships,publication_year,cited_by_count,abstract_inverted_index,primary_location"
            data = await _fetch_json(url)

            existing_titles = {p["title"].lower()[:50] for p in papers}
            for w in (data.get("results") or []):
                title = w.get("title", "")
                if title and title.lower()[:50] not in existing_titles:
                    abstract = ""
                    inv = w.get("abstract_inverted_index")
                    if inv:
                        wps = sorted([(pos, wd) for wd, poss in inv.items() for pos in poss])
                        abstract = " ".join(wd for _, wd in wps)[:300]

                    authors = [
                        (a.get("author") or {}).get("display_name", "")
                        for a in (w.get("authorships") or [])[:4]
                    ]
                    doi = (w.get("doi") or "").replace("https://doi.org/", "")
                    loc = w.get("primary_location") or {}
                    paper_url = loc.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else "")

                    papers.append({
                        "title": title,
                        "authors": [a for a in authors if a],
                        "year": w.get("publication_year"),
                        "citations": w.get("cited_by_count", 0),
                        "abstract": abstract,
                        "url": paper_url,
                        "doi": doi,
                        "source": "openalex",
                    })
        except Exception as e:
            print(f"OpenAlex concept search error: {e}")

    return papers


async def search_papers(query, limit=5):
    """Smart paper search — extracts keywords, searches both sources in parallel."""
    extracted = extract_keywords(query)
    print(f"Paper search — query: '{query}' → keywords: {extracted['keywords']}")

    ss_task = search_semantic_scholar(query, extracted["keywords"], limit)
    oa_task = search_openalex(query, extracted["keywords"], limit)

    ss_results, oa_results = await asyncio.gather(ss_task, oa_task)

    # Merge and deduplicate
    all_papers = []
    seen_titles = set()

    for papers in [ss_results, oa_results]:
        for p in papers:
            if not p.get("title"):
                continue
            title_key = p["title"].lower().strip()[:60]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                all_papers.append(p)

    # Sort by citations
    all_papers.sort(key=lambda x: x.get("citations") or 0, reverse=True)

    return all_papers[:8]


async def get_citation_network(paper_id, source="semantic_scholar", depth=1):
    """Fetch citations and references for a paper from Semantic Scholar."""
    base = "https://api.semanticscholar.org/graph/v1/paper"
    fields = "title,authors,year,citationCount,url,externalIds"

    async def fetch_json(url):
        try:
            loop = asyncio.get_event_loop()
            import urllib.request
            def do_fetch():
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "Meridian/1.0")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode())
            return await asyncio.to_thread(do_fetch)
        except Exception:
            return None

    # Fetch the root paper
    root_url = f"{base}/{paper_id}?fields={fields},citations.{fields},references.{fields}"
    data = await fetch_json(root_url)
    if not data:
        return {"nodes": [], "links": []}

    nodes = {}
    links = []

    # Root node
    root_id = data.get("paperId", paper_id)
    nodes[root_id] = {
        "id": root_id,
        "title": data.get("title", ""),
        "year": data.get("year"),
        "citations": data.get("citationCount", 0),
        "authors": [a.get("name", "") for a in (data.get("authors") or [])[:3]],
        "url": data.get("url", ""),
        "type": "root",
    }

    # Citations (papers that cite this one)
    for ref in (data.get("citations") or [])[:15]:
        pid = ref.get("paperId")
        if not pid:
            continue
        nodes[pid] = {
            "id": pid,
            "title": ref.get("title", ""),
            "year": ref.get("year"),
            "citations": ref.get("citationCount", 0),
            "authors": [a.get("name", "") for a in (ref.get("authors") or [])[:3]],
            "url": ref.get("url", ""),
            "type": "citing",
        }
        links.append({"source": pid, "target": root_id, "type": "cites"})

    # References (papers this one cites)
    for ref in (data.get("references") or [])[:15]:
        pid = ref.get("paperId")
        if not pid:
            continue
        if pid not in nodes:
            nodes[pid] = {
                "id": pid,
                "title": ref.get("title", ""),
                "year": ref.get("year"),
                "citations": ref.get("citationCount", 0),
                "authors": [a.get("name", "") for a in (ref.get("authors") or [])[:3]],
                "url": ref.get("url", ""),
                "type": "referenced",
            }
        links.append({"source": root_id, "target": pid, "type": "cites"})

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "root_id": root_id,
    }
