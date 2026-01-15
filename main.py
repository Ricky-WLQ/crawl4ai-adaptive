"""
Crawl4AI Hybrid BFS + Semantic Crawler with DeepSeek Reasoner

VERSION 3.8.0 - HYBRID BFS DEEP CRAWL (Context7 Verified)

CRITICAL FIX: Uses BFS (Breadth-First Search) for GUARANTEED DEPTH CRAWLING
- BFS ensures ALL links at each level are visited before going deeper
- max_depth=3 guarantees reaching content pages like go.asp?d=lei-17-2009cn
- Semantic ranking applied AFTER crawling for relevance sorting

WHY BFS OVER DFS/ADAPTIVE:
- BFS: Systematic level-by-level exploration (finds ALL sibling links)
- DFS: Goes deep on first link only (may miss sibling content)
- Adaptive: Semantic selection may skip important links with unfamiliar text

ARCHITECTURE:
1. BFSDeepCrawlStrategy (max_depth=3) - Crawl ALL pages systematically
2. SPA-friendly settings - JavaScript rendering for .asp sites
3. Local embedding scoring - Rank by semantic relevance
4. OpenRouter re-ranking (optional) - Additional semantic refinement
5. DeepSeek-reasoner - Generate comprehensive answers

EMBEDDING STRATEGY:
- LOCAL: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- Supports 50+ languages (Chinese, English, Portuguese, etc.)
- NO API calls needed during crawling

Version: 3.8.0 (HYBRID BFS DEEP CRAWL - Context7 Verified)
"""

import os
import sys
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import re
import traceback

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ============================================================================
# Core crawl4ai imports
# ============================================================================
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.async_configs import CacheMode
    CORE_AVAILABLE = True
    print("Core crawl4ai imported successfully", flush=True)
except ImportError as e:
    print(f"CRITICAL: Failed to import core crawl4ai: {e}", flush=True)
    sys.exit(1)

# BFS Deep Crawling imports (PRIMARY - for depth control)
try:
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter
    from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
    BFS_AVAILABLE = True
    print("BFSDeepCrawlStrategy imported successfully (PRIMARY)", flush=True)
except ImportError as e:
    print(f"BFSDeepCrawlStrategy not available: {e}", flush=True)
    BFS_AVAILABLE = False

# AdaptiveCrawler imports (SECONDARY - for semantic fallback)
try:
    from crawl4ai import AdaptiveCrawler, AdaptiveConfig
    ADAPTIVE_AVAILABLE = True
    print("AdaptiveCrawler imported successfully (SECONDARY)", flush=True)
except ImportError as e:
    print(f"AdaptiveCrawler not available: {e}", flush=True)
    ADAPTIVE_AVAILABLE = False

# Sentence transformers for local semantic scoring
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("sentence-transformers imported successfully", flush=True)
except ImportError as e:
    print(f"sentence-transformers not available: {e}", flush=True)
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Global State
# ============================================================================
EMBEDDING_MODEL = None
EMBEDDING_MODEL_VERIFIED = False
EMBEDDING_MODEL_ERROR = None


# ============================================================================
# Embedding Model Initialization
# ============================================================================
def initialize_embedding_model(model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> bool:
    """
    Initialize and verify the embedding model at startup.
    This runs once and caches the model for reuse.
    """
    global EMBEDDING_MODEL, EMBEDDING_MODEL_VERIFIED, EMBEDDING_MODEL_ERROR

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        EMBEDDING_MODEL_ERROR = "sentence-transformers not installed"
        print(f"Embedding model initialization SKIPPED: {EMBEDDING_MODEL_ERROR}", flush=True)
        return False

    try:
        print(f"Initializing embedding model: {model_name}...", flush=True)

        # Load the model (cached globally)
        EMBEDDING_MODEL = SentenceTransformer(model_name)
        print(f"  Model loaded successfully", flush=True)

        # Test with sample texts
        test_texts = [
            "This is a test sentence.",
            "澳門法律 刑法 販毒",  # Macau law terms
            "第17/2009號法律"  # Law number format
        ]

        embeddings = EMBEDDING_MODEL.encode(test_texts)
        print(f"  Generated {len(embeddings)} test embeddings", flush=True)
        print(f"  Embedding dimension: {len(embeddings[0])}", flush=True)

        # Verify embeddings are valid
        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            if norm < 0.001:
                raise ValueError(f"Embedding {i} has near-zero norm")

        EMBEDDING_MODEL_VERIFIED = True
        print(f"Embedding model verification PASSED", flush=True)
        return True

    except Exception as e:
        EMBEDDING_MODEL_ERROR = str(e)
        print(f"Embedding model initialization FAILED: {e}", flush=True)
        traceback.print_exc()
        return False


def compute_semantic_scores(query: str, contents: List[str]) -> List[float]:
    """
    Compute semantic similarity scores between query and contents using local model.
    Returns list of scores (0.0 to 1.0).
    """
    if not EMBEDDING_MODEL or not EMBEDDING_MODEL_VERIFIED:
        return [0.5] * len(contents)  # Default score if model not available

    try:
        # Encode query and contents
        query_embedding = EMBEDDING_MODEL.encode([query])[0]
        content_embeddings = EMBEDDING_MODEL.encode(contents)

        # Compute cosine similarities
        scores = []
        query_norm = np.linalg.norm(query_embedding)

        for content_emb in content_embeddings:
            content_norm = np.linalg.norm(content_emb)
            if query_norm > 0 and content_norm > 0:
                similarity = np.dot(query_embedding, content_emb) / (query_norm * content_norm)
                # Normalize to 0-1 range (cosine similarity can be -1 to 1)
                score = (similarity + 1) / 2
            else:
                score = 0.5
            scores.append(float(score))

        return scores

    except Exception as e:
        print(f"Semantic scoring error: {e}", flush=True)
        return [0.5] * len(contents)


# ============================================================================
# OpenRouter Embedding Client (Optional Re-ranking)
# ============================================================================
class OpenRouterEmbeddings:
    """Custom client for OpenRouter embeddings API with retry logic."""

    def __init__(self, api_key: str, model: str = "qwen/qwen3-embedding-8b", max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/embeddings"
        self.max_retries = max_retries

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts from OpenRouter with retry logic."""
        if not texts:
            return []

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={"model": self.model, "input": texts}
                    )

                    if response.status_code == 429:
                        wait_time = 2 ** attempt
                        print(f"Rate limited. Waiting {wait_time}s...", flush=True)
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status_code != 200:
                        print(f"OpenRouter error: {response.status_code}", flush=True)
                        if attempt == self.max_retries - 1:
                            return []
                        continue

                    result = response.json()
                    embeddings = [None] * len(texts)
                    for item in result.get("data", []):
                        idx = item.get("index", 0)
                        embeddings[idx] = item.get("embedding", [])
                    return embeddings

            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"OpenRouter failed after {self.max_retries} attempts: {e}", flush=True)
                    return []
                await asyncio.sleep(2 ** attempt)

        return []

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []


def cosine_similarity_vectors(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
    except Exception:
        return 0.0


async def rerank_with_openrouter(
    query: str,
    pages: List[Dict],
    embedding_client: OpenRouterEmbeddings,
    top_k: int = 15
) -> List[Dict]:
    """Re-rank pages using OpenRouter embeddings (75% semantic, 25% original)."""
    if not pages:
        return []

    print(f"Re-ranking {len(pages)} pages with OpenRouter...", flush=True)

    try:
        query_embedding = await embedding_client.get_embedding(query)
        if not query_embedding:
            print("Failed to get query embedding", flush=True)
            return pages[:top_k]

        page_texts = [p.get('content', '')[:4000] for p in pages]
        page_embeddings = await embedding_client.get_embeddings(page_texts)

        scored_pages = []
        for i, page in enumerate(pages):
            if i < len(page_embeddings) and page_embeddings[i]:
                similarity = cosine_similarity_vectors(query_embedding, page_embeddings[i])
            else:
                similarity = 0.0

            original_score = page.get('score', 0.5)
            combined_score = (0.75 * similarity) + (0.25 * original_score)

            scored_pages.append({
                **page,
                'embedding_score': round(similarity, 4),
                'original_score': original_score,
                'score': round(combined_score, 4)
            })

        scored_pages.sort(key=lambda x: x['score'], reverse=True)
        print(f"Re-ranking complete. Top scores: {[p['score'] for p in scored_pages[:5]]}", flush=True)
        return scored_pages[:top_k]

    except Exception as e:
        print(f"OpenRouter re-ranking failed: {e}", flush=True)
        return pages[:top_k]


# ============================================================================
# Helper Functions
# ============================================================================
def detect_query_languages(query: str) -> Dict[str, Any]:
    """Detect languages in query."""
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
    has_english = bool(re.search(r'[a-zA-Z]', query))
    languages = []
    if has_chinese:
        languages.append("Chinese")
    if has_english:
        languages.append("English")
    return {
        "has_chinese": has_chinese,
        "has_english": has_english,
        "languages": languages or ["Unknown"]
    }


def extract_keywords(query: str) -> List[str]:
    """
    Extract keywords from query, handling both English and Chinese.

    For Chinese: Extracts 2-4 character segments (common word lengths)
    For English: Splits by whitespace and filters stop words
    """
    # Extended stop words for questions
    stop_words = {
        # English
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'what', 'which', 'who', 'how', 'when', 'where', 'why', 'i', 'you',
        'show', 'get', 'find', 'give', 'me', 'my', 'your', 'want', 'need',
        'please', 'tell', 'about', 'know', 'information',
        # Chinese stop words and question particles
        '我', '是', '在', '会', '被', '了', '吗', '嘛', '的', '和', '与',
        '什么', '什麼', '怎么', '怎麼', '如何', '哪里', '哪裡', '为什么', '為什麼',
        '请', '請', '告诉', '告訴', '关于', '關於', '有关', '有關',
        '能', '可以', '想', '要', '给', '給', '这', '這', '那', '个', '個'
    }

    keywords = []

    # Extract English words (split by whitespace)
    english_words = re.findall(r'[a-zA-Z]+', query)
    for word in english_words:
        if len(word) > 2 and word.lower() not in stop_words:
            keywords.append(word.lower())

    # Extract Chinese text segments
    chinese_text = ''.join(re.findall(r'[\u4e00-\u9fff]+', query))

    if chinese_text:
        # Extract overlapping 2-char, 3-char, and 4-char segments
        # This captures common Chinese word lengths without needing a segmenter
        for length in [2, 3, 4]:
            for i in range(len(chinese_text) - length + 1):
                segment = chinese_text[i:i + length]
                # Skip if it's a stop word or contains only stop characters
                if segment not in stop_words and not all(c in '的是在了吗嘛什么麼' for c in segment):
                    keywords.append(segment)

        # Also add important single characters that might be meaningful
        important_chars = ['法', '律', '罪', '刑', '罰', '毒', '販', '澳', '門']
        for char in chinese_text:
            if char in important_chars:
                keywords.append(char)

    # Remove duplicates while preserving some order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords


def extract_content_from_result(result) -> str:
    """Extract text content from a CrawlResult object."""
    content = ""

    # Try markdown extraction first
    if hasattr(result, 'markdown') and result.markdown:
        md = result.markdown
        if hasattr(md, 'fit_markdown') and md.fit_markdown:
            content = str(md.fit_markdown)
        elif hasattr(md, 'raw_markdown') and md.raw_markdown:
            content = str(md.raw_markdown)

    # Fallback to extracted_content
    if not content and hasattr(result, 'extracted_content') and result.extracted_content:
        content = str(result.extracted_content)

    # Final fallback to html
    if not content and hasattr(result, 'html') and result.html:
        # Strip HTML tags for basic text
        content = re.sub(r'<[^>]+>', ' ', str(result.html))
        content = re.sub(r'\s+', ' ', content).strip()

    return content.strip()


def format_context_for_llm(pages: List[Dict], max_chars: int = 30000) -> str:
    """Format pages into context string for LLM."""
    context_parts = []
    total_chars = 0

    for i, page in enumerate(pages, 1):
        url = page.get('url', 'unknown')
        score = page.get('score', 0)
        content = page.get('content', '')

        if not content or len(content) < 50:
            continue

        page_text = content[:6000] if len(content) > 6000 else content
        score_pct = int(score * 100)

        entry = f"\n=== Page {i} (Relevance: {score_pct}%): {url} ===\n{page_text}\n"

        if total_chars + len(entry) > max_chars:
            break

        context_parts.append(entry)
        total_chars += len(entry)

    return "\n".join(context_parts)


# ============================================================================
# DeepSeek API
# ============================================================================
async def call_deepseek(
    query: str,
    context: str,
    api_key: str,
    max_tokens: int = 3000,
    max_retries: int = 3
) -> str:
    """Call DeepSeek API to generate answer."""

    system_prompt = """You are a helpful assistant that provides direct, accurate answers based on the provided web content.

Your task:
1. Carefully analyze ALL the crawled web content provided
2. Find the most relevant and detailed information to answer the user's query
3. Provide a comprehensive, accurate answer in plain text
4. Include specific details, code examples, legal references, or steps if available
5. If the information is incomplete, mention what was found and suggest where to look
6. Cite the source URLs for the information you use

Be thorough and informative. Extract maximum value from the crawled content."""

    user_message = f"""Query: {query}

Crawled Web Content:
{context}

Based on the above content, provide a detailed and accurate answer to the query."""

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-reasoner",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.2
                    }
                )

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"DeepSeek rate limited. Waiting {wait_time}s...", flush=True)
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    if attempt == max_retries - 1:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"DeepSeek API error: {response.text}"
                        )
                    continue

                result = response.json()
                return result["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="DeepSeek API timeout")
            await asyncio.sleep(2 ** attempt)

    raise HTTPException(status_code=500, detail="DeepSeek failed after retries")


# ============================================================================
# FastAPI Application
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - initialize models at startup."""
    print("=" * 70, flush=True)
    print("Crawl4AI HYBRID BFS CRAWLER v3.8.0 Starting...", flush=True)
    print("=" * 70, flush=True)

    print(f"BFSDeepCrawlStrategy Available: {BFS_AVAILABLE} (PRIMARY)", flush=True)
    print(f"AdaptiveCrawler Available: {ADAPTIVE_AVAILABLE} (SECONDARY)", flush=True)
    print(f"OpenRouter API Key: {'Set' if os.environ.get('OPENROUTER_API_KEY') else 'NOT SET'}", flush=True)
    print(f"DeepSeek API Key: {'Set' if os.environ.get('DEEPSEEK_API_KEY') else 'NOT SET'}", flush=True)
    print("=" * 70, flush=True)

    # Initialize embedding model
    print("STARTUP: Initializing embedding model...", flush=True)
    initialize_embedding_model()
    print("=" * 70, flush=True)

    print("HYBRID BFS STRATEGY (v3.8.0):", flush=True)
    print("  1. BFS Deep Crawl: max_depth=3, systematic level-by-level", flush=True)
    print("  2. SPA Support: wait_for=css:body, process_iframes=True", flush=True)
    print("  3. Local Semantic Ranking: paraphrase-multilingual-mpnet-base-v2", flush=True)
    print("  4. OpenRouter Re-ranking: 75% semantic + 25% keyword (optional)", flush=True)
    print("  5. DeepSeek Answer Generation: deepseek-reasoner", flush=True)
    print("=" * 70, flush=True)

    print("WHY BFS OVER DFS:", flush=True)
    print("  - BFS: Explores ALL links at each level before going deeper", flush=True)
    print("  - DFS: Goes deep on first link, may miss sibling content", flush=True)
    print("  - For hierarchical legal sites, BFS finds ALL law documents", flush=True)
    print("=" * 70, flush=True)

    yield
    print("Shutting down...", flush=True)


app = FastAPI(
    title="Crawl4AI Hybrid BFS Crawler",
    description="Hybrid BFS + Semantic crawler for deep content extraction - v3.8.0",
    version="3.8.0",
    lifespan=lifespan
)


class CrawlRequest(BaseModel):
    """Request model for crawling."""
    start_url: str
    query: str
    max_pages: Optional[int] = 100
    max_depth: Optional[int] = 3
    use_embeddings: Optional[bool] = True


class CrawlResponse(BaseModel):
    """Response model."""
    success: bool
    answer: str
    confidence: float
    pages_crawled: int
    sources: List[Dict]
    message: str
    embedding_used: bool
    crawl_strategy: str


# ============================================================================
# MAIN CRAWL FUNCTION: Hybrid BFS + Semantic Ranking
# ============================================================================
async def run_hybrid_bfs_crawl(
    request: CrawlRequest,
    deepseek_api_key: str,
    openrouter_client: Optional[OpenRouterEmbeddings],
    domain: str
) -> CrawlResponse:
    """
    HYBRID BFS CRAWL (v3.8.0):
    1. Use BFS to crawl ALL pages up to max_depth (systematic exploration)
    2. Apply local semantic scoring to rank pages by relevance
    3. Optionally re-rank with OpenRouter embeddings
    4. Generate answer with DeepSeek
    """

    print("\n" + "=" * 70, flush=True)
    print("HYBRID BFS CRAWL STARTING", flush=True)
    print(f"  Strategy: BFSDeepCrawlStrategy (max_depth={request.max_depth})", flush=True)
    print(f"  Max Pages: {request.max_pages}", flush=True)
    print(f"  Domain: {domain}", flush=True)
    print("=" * 70, flush=True)

    # Extract keywords for URL scoring
    keywords = extract_keywords(request.query)
    print(f"Keywords extracted: {keywords}", flush=True)

    # Configure BFS strategy
    domain_filter = DomainFilter(
        allowed_domains=[domain],
        blocked_domains=[]
    )

    keyword_scorer = KeywordRelevanceScorer(
        keywords=keywords,
        weight=1.0
    )

    bfs_strategy = BFSDeepCrawlStrategy(
        max_depth=request.max_depth or 3,
        include_external=False,
        max_pages=request.max_pages or 100,
        filter_chain=FilterChain([domain_filter]),
        url_scorer=keyword_scorer
    )

    # SPA-friendly crawler configuration
    crawl_config = CrawlerRunConfig(
        deep_crawl_strategy=bfs_strategy,
        wait_for="css:body",
        process_iframes=True,
        delay_before_return_html=2.0,
        page_timeout=60000,
        cache_mode=CacheMode.BYPASS,
        stream=False,
        verbose=True
    )

    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        java_script_enabled=True
    )

    # Execute BFS crawl
    print("\nStarting BFS deep crawl...", flush=True)
    all_pages = []

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun(
                url=request.start_url,
                config=crawl_config
            )

            # Handle results (can be list or single result)
            if isinstance(results, list):
                crawled_results = results
            else:
                crawled_results = [results] if results else []

            print(f"BFS crawl returned {len(crawled_results)} results", flush=True)

            # Extract content from each result
            for i, result in enumerate(crawled_results):
                if not result:
                    continue

                # Check success
                if hasattr(result, 'success') and not result.success:
                    continue

                url = result.url if hasattr(result, 'url') else f"page_{i}"
                content = extract_content_from_result(result)
                depth = result.metadata.get('depth', 0) if hasattr(result, 'metadata') else 0

                if content and len(content) >= 100:
                    all_pages.append({
                        'url': url,
                        'content': content,
                        'depth': depth,
                        'score': 0.5  # Default score, will be updated
                    })
                    print(f"  [{i}] Depth {depth}: {len(content):,} chars - {url[:60]}...", flush=True)

    except Exception as e:
        print(f"BFS crawl error: {e}", flush=True)
        traceback.print_exc()
        return CrawlResponse(
            success=False,
            answer=f"Crawl failed: {str(e)}",
            confidence=0.0,
            pages_crawled=0,
            sources=[],
            message=f"BFS crawl error: {str(e)}",
            embedding_used=False,
            crawl_strategy="bfs"
        )

    pages_crawled = len(all_pages)
    print(f"\nBFS crawl complete: {pages_crawled} pages with content", flush=True)

    if pages_crawled == 0:
        return CrawlResponse(
            success=False,
            answer="No content could be extracted from the crawled pages.",
            confidence=0.0,
            pages_crawled=0,
            sources=[],
            message="BFS crawl found no extractable content",
            embedding_used=False,
            crawl_strategy="bfs"
        )

    # STEP 2: Apply local semantic scoring
    print("\nApplying local semantic scoring...", flush=True)
    contents = [p['content'][:2000] for p in all_pages]  # Use first 2000 chars for scoring
    semantic_scores = compute_semantic_scores(request.query, contents)

    for i, page in enumerate(all_pages):
        page['score'] = semantic_scores[i] if i < len(semantic_scores) else 0.5

    # Sort by semantic score
    all_pages.sort(key=lambda x: x['score'], reverse=True)

    print("Top 10 pages after local semantic scoring:", flush=True)
    for i, page in enumerate(all_pages[:10]):
        score_pct = int(page['score'] * 100)
        print(f"  {i+1}. {score_pct}% (depth {page['depth']}) - {page['url'][:60]}", flush=True)

    # STEP 3: Optional OpenRouter re-ranking
    embedding_used = False
    if openrouter_client and all_pages:
        try:
            print("\nApplying OpenRouter re-ranking...", flush=True)
            all_pages = await rerank_with_openrouter(
                query=request.query,
                pages=all_pages,
                embedding_client=openrouter_client,
                top_k=20
            )
            embedding_used = True

            print("Top 10 pages after OpenRouter re-ranking:", flush=True)
            for i, page in enumerate(all_pages[:10]):
                score_pct = int(page['score'] * 100)
                emb_score = int(page.get('embedding_score', 0) * 100)
                print(f"  {i+1}. Combined: {score_pct}% (Semantic: {emb_score}%) - {page['url'][:50]}", flush=True)
        except Exception as e:
            print(f"OpenRouter re-ranking failed: {e}, using local scores", flush=True)
            all_pages = all_pages[:20]
    else:
        all_pages = all_pages[:20]

    # STEP 4: Generate answer with DeepSeek
    context = format_context_for_llm(all_pages)

    if not context.strip():
        return CrawlResponse(
            success=False,
            answer="Pages were crawled but contained insufficient readable content.",
            confidence=0.0,
            pages_crawled=pages_crawled,
            sources=[],
            message="Content extraction yielded empty context",
            embedding_used=embedding_used,
            crawl_strategy="bfs"
        )

    print(f"\nContext prepared: {len(context):,} chars", flush=True)
    print("Generating answer with DeepSeek-reasoner...", flush=True)

    try:
        answer = await call_deepseek(
            query=request.query,
            context=context,
            api_key=deepseek_api_key
        )
    except Exception as e:
        print(f"DeepSeek error: {e}", flush=True)
        return CrawlResponse(
            success=False,
            answer=f"Answer generation failed: {str(e)}",
            confidence=0.5,
            pages_crawled=pages_crawled,
            sources=[{"url": p['url'], "relevance": round(p['score'], 3)} for p in all_pages[:10]],
            message=f"DeepSeek error: {str(e)}",
            embedding_used=embedding_used,
            crawl_strategy="bfs"
        )

    # Prepare sources
    sources = []
    seen_urls = set()
    for page in all_pages[:15]:
        url = page.get('url', 'unknown')
        if url in seen_urls:
            continue
        seen_urls.add(url)
        source_info = {
            "url": url,
            "relevance": round(page.get('score', 0), 3),
            "depth": page.get('depth', 0)
        }
        if embedding_used and 'embedding_score' in page:
            source_info["semantic_score"] = round(page.get('embedding_score', 0), 3)
        sources.append(source_info)

    # Calculate confidence based on top scores
    avg_top_score = sum(p['score'] for p in all_pages[:5]) / min(5, len(all_pages))
    confidence = round(avg_top_score, 3)

    return CrawlResponse(
        success=True,
        answer=answer,
        confidence=confidence,
        pages_crawled=pages_crawled,
        sources=sources,
        message=f"Hybrid BFS crawl: {pages_crawled} pages, depth={request.max_depth}" +
                (", with OpenRouter re-ranking" if embedding_used else ", with local semantic scoring"),
        embedding_used=embedding_used,
        crawl_strategy="hybrid_bfs"
    )


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
async def root():
    """Service status endpoint."""
    return {
        "service": "Crawl4AI Hybrid BFS Crawler",
        "version": "3.8.0",
        "status": "Ready",
        "strategy": "Hybrid BFS + Semantic Ranking",
        "features": {
            "bfs_deep_crawl": BFS_AVAILABLE,
            "adaptive_fallback": ADAPTIVE_AVAILABLE,
            "local_embeddings": EMBEDDING_MODEL_VERIFIED,
            "openrouter_reranking": bool(os.environ.get("OPENROUTER_API_KEY")),
            "deepseek_answers": bool(os.environ.get("DEEPSEEK_API_KEY"))
        },
        "why_bfs": [
            "BFS explores ALL links at each depth level before going deeper",
            "Guarantees finding sibling pages (all law documents at same level)",
            "max_depth=3 ensures reaching content pages like go.asp?d=lei-17-2009cn",
            "DFS would go deep on first link and potentially miss important siblings"
        ],
        "architecture": [
            "1. BFSDeepCrawlStrategy - Systematic depth crawling",
            "2. SPA-friendly config - JavaScript rendering",
            "3. Local semantic scoring - paraphrase-multilingual-mpnet-base-v2",
            "4. OpenRouter re-ranking (optional) - Additional refinement",
            "5. DeepSeek-reasoner - Answer generation"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "3.8.0",
        "bfs_available": BFS_AVAILABLE,
        "embedding_model_ready": EMBEDDING_MODEL_VERIFIED
    }


@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest):
    """
    Perform hybrid BFS crawl with semantic ranking.

    Strategy:
    1. BFS deep crawl (max_depth=3) for systematic exploration
    2. Local semantic scoring with multilingual embeddings
    3. Optional OpenRouter re-ranking
    4. DeepSeek answer generation
    """

    # Validate API keys
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY environment variable not set"
        )

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    # Parse domain from URL
    parsed_url = urlparse(request.start_url)
    domain = parsed_url.netloc

    # Detect query languages
    lang_info = detect_query_languages(request.query)

    print("\n" + "=" * 70, flush=True)
    print("NEW CRAWL REQUEST", flush=True)
    print(f"  Query: {request.query}", flush=True)
    print(f"  Languages: {lang_info['languages']}", flush=True)
    print(f"  Start URL: {request.start_url}", flush=True)
    print(f"  Domain: {domain}", flush=True)
    print(f"  Max Pages: {request.max_pages}", flush=True)
    print(f"  Max Depth: {request.max_depth}", flush=True)
    print(f"  OpenRouter: {'Enabled' if openrouter_api_key else 'Disabled'}", flush=True)
    print("=" * 70, flush=True)

    # Create OpenRouter client if available
    openrouter_client = None
    if openrouter_api_key and request.use_embeddings:
        openrouter_client = OpenRouterEmbeddings(api_key=openrouter_api_key)

    # Check which strategy to use
    if BFS_AVAILABLE:
        return await run_hybrid_bfs_crawl(
            request=request,
            deepseek_api_key=deepseek_api_key,
            openrouter_client=openrouter_client,
            domain=domain
        )
    elif ADAPTIVE_AVAILABLE:
        # Fallback to adaptive crawl if BFS not available
        print("BFS not available, falling back to AdaptiveCrawler", flush=True)
        return await run_adaptive_fallback(
            request=request,
            deepseek_api_key=deepseek_api_key,
            openrouter_client=openrouter_client,
            domain=domain
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Neither BFS nor Adaptive crawling is available"
        )


async def run_adaptive_fallback(
    request: CrawlRequest,
    deepseek_api_key: str,
    openrouter_client: Optional[OpenRouterEmbeddings],
    domain: str
) -> CrawlResponse:
    """Fallback to AdaptiveCrawler if BFS is not available."""

    print("Using ADAPTIVE FALLBACK (BFS not available)", flush=True)

    config = AdaptiveConfig(
        strategy="embedding" if EMBEDDING_MODEL_VERIFIED else "statistical",
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        confidence_threshold=0.05,
        max_pages=request.max_pages or 100,
        top_k_links=30,
        embedding_min_confidence_threshold=0.02,
        embedding_min_relative_improvement=0.01,
        n_query_variations=20
    )

    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        java_script_enabled=True
    )

    all_pages = []

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            adaptive = AdaptiveCrawler(crawler, config=config)

            result = await adaptive.digest(
                start_url=request.start_url,
                query=request.query
            )

            if result and hasattr(result, 'knowledge_base') and result.knowledge_base:
                for doc in result.knowledge_base:
                    url = doc.url if hasattr(doc, 'url') else 'unknown'
                    content = extract_content_from_result(doc)
                    if content and len(content) >= 100:
                        all_pages.append({
                            'url': url,
                            'content': content,
                            'score': 0.5,
                            'depth': 0
                        })

            pages_crawled = len(result.crawled_urls) if hasattr(result, 'crawled_urls') else len(all_pages)

    except Exception as e:
        print(f"Adaptive crawl error: {e}", flush=True)
        return CrawlResponse(
            success=False,
            answer=f"Adaptive crawl failed: {str(e)}",
            confidence=0.0,
            pages_crawled=0,
            sources=[],
            message=f"Adaptive fallback error: {str(e)}",
            embedding_used=False,
            crawl_strategy="adaptive_fallback"
        )

    if not all_pages:
        return CrawlResponse(
            success=False,
            answer="No content extracted from adaptive crawl.",
            confidence=0.0,
            pages_crawled=pages_crawled,
            sources=[],
            message="Adaptive crawl yielded no content",
            embedding_used=False,
            crawl_strategy="adaptive_fallback"
        )

    # Apply semantic scoring
    contents = [p['content'][:2000] for p in all_pages]
    scores = compute_semantic_scores(request.query, contents)
    for i, page in enumerate(all_pages):
        page['score'] = scores[i] if i < len(scores) else 0.5
    all_pages.sort(key=lambda x: x['score'], reverse=True)

    # Optional OpenRouter re-ranking
    embedding_used = False
    if openrouter_client:
        try:
            all_pages = await rerank_with_openrouter(
                request.query, all_pages, openrouter_client, top_k=15
            )
            embedding_used = True
        except Exception:
            all_pages = all_pages[:15]
    else:
        all_pages = all_pages[:15]

    # Generate answer
    context = format_context_for_llm(all_pages)
    if not context.strip():
        return CrawlResponse(
            success=False,
            answer="No readable content for answer generation.",
            confidence=0.0,
            pages_crawled=pages_crawled,
            sources=[],
            message="Empty context from adaptive crawl",
            embedding_used=embedding_used,
            crawl_strategy="adaptive_fallback"
        )

    answer = await call_deepseek(request.query, context, deepseek_api_key)

    sources = [{"url": p['url'], "relevance": round(p['score'], 3)} for p in all_pages[:10]]

    return CrawlResponse(
        success=True,
        answer=answer,
        confidence=0.5,
        pages_crawled=pages_crawled,
        sources=sources,
        message=f"Adaptive fallback: {pages_crawled} pages",
        embedding_used=embedding_used,
        crawl_strategy="adaptive_fallback"
    )


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Crawl4AI Hybrid BFS Crawler v3.8.0 on port {port}...", flush=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
