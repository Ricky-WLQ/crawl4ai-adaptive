"""
Crawl4AI Deep Crawler with DeepSeek
- Deep crawling with DFS strategy and domain restriction
- Goes deep into one path before backtracking
- Stays on the same site
- DeepSeek-reasoner for answer generation
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, List
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Test imports at startup
try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
    print("AsyncWebCrawler imported successfully", flush=True)
except ImportError as e:
    print(f"Failed to import AsyncWebCrawler: {e}", flush=True)
    sys.exit(1)

try:
    from crawl4ai.deep_crawling import DFSDeepCrawlStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, DomainFilter
    DEEP_CRAWL_AVAILABLE = True
    print("Deep crawling components imported successfully", flush=True)
except ImportError as e:
    print(f"Deep crawling not available: {e}", flush=True)
    DEEP_CRAWL_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    print("=" * 50, flush=True)
    print("Crawl4AI Deep Crawler Starting...", flush=True)
    print(f"Deep Crawling Available: {DEEP_CRAWL_AVAILABLE}", flush=True)
    print("Strategy: DFS with domain restriction", flush=True)
    print("Answer: DeepSeek-reasoner", flush=True)
    print("=" * 50, flush=True)
    yield
    print("Shutting down...", flush=True)


app = FastAPI(
    title="Crawl4AI Deep Crawler",
    description="Deep web crawler with domain restriction and DeepSeek reasoning",
    version="1.0.0",
    lifespan=lifespan
)


class CrawlRequest(BaseModel):
    """Request model for deep crawling."""
    start_url: str
    query: str
    max_depth: Optional[int] = 3
    max_pages: Optional[int] = 10


class CrawlResponse(BaseModel):
    """Response model with direct answer."""
    success: bool
    answer: str
    pages_crawled: int
    sources: List[dict]
    message: str


async def call_deepseek(
    query: str,
    context: str,
    api_key: str,
    max_tokens: int = 2000
) -> str:
    """Call DeepSeek API to generate an answer."""

    system_prompt = """You are a helpful assistant that provides direct, accurate answers based on the provided web content.

Your task:
1. Carefully analyze ALL the crawled web content provided
2. Find the most relevant and detailed information to answer the user's query
3. Provide a comprehensive, accurate answer in plain text
4. Include specific details, code examples, or steps if available in the content
5. If the information is incomplete, mention what was found and suggest where to look
6. Cite the source URLs for the information you use

Be thorough and informative. Extract maximum value from the crawled content."""

    user_message = f"""Query: {query}

Crawled Web Content:
{context}

Based on the above content, provide a detailed and accurate answer to the query."""

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

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"DeepSeek API error: {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]


def format_context_for_llm(results: List, max_chars: int = 20000) -> str:
    """Format crawled pages into context for LLM."""
    context_parts = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        url = result.url if hasattr(result, 'url') else str(result)

        # Get markdown content (cleaner than raw HTML)
        content = ""
        if hasattr(result, 'markdown') and result.markdown:
            content = result.markdown
        elif hasattr(result, 'text') and result.text:
            content = result.text
        elif hasattr(result, 'html') and result.html:
            content = result.html[:5000]

        if not content or len(content) < 50:
            continue

        # Truncate individual page content
        page_text = content[:4000] if len(content) > 4000 else content

        entry = f"""
=== Page {i}: {url} ===
{page_text}

"""
        if total_chars + len(entry) > max_chars:
            break

        context_parts.append(entry)
        total_chars += len(entry)

    return "\n".join(context_parts)


@app.get("/")
async def root():
    """Service status endpoint."""
    return {
        "message": "Crawl4AI Deep Crawler is running!",
        "version": "1.0.0",
        "deep_crawl_available": DEEP_CRAWL_AVAILABLE,
        "features": [
            "DFS deep crawling strategy",
            "Domain restriction (stays on same site)",
            "DeepSeek-reasoner for answer generation"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/crawl", response_model=CrawlResponse)
async def deep_crawl(request: CrawlRequest):
    """
    Perform deep crawling and return a direct answer.

    - Stays on the same domain (no external links)
    - Uses DFS to go deep into one path before backtracking
    - DeepSeek-reasoner generates answer from crawled content
    """

    if not DEEP_CRAWL_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Deep crawling not available. Check crawl4ai installation."
        )

    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY environment variable not set"
        )

    try:
        # Extract domain from start URL
        parsed_url = urlparse(request.start_url)
        domain = parsed_url.netloc

        print(f"\n{'='*50}", flush=True)
        print(f"Query: {request.query}", flush=True)
        print(f"Start URL: {request.start_url}", flush=True)
        print(f"Domain: {domain} (external links blocked)", flush=True)
        print(f"Max depth: {request.max_depth}, Max pages: {request.max_pages}", flush=True)
        print(f"{'='*50}", flush=True)

        # Create domain filter to stay on same site
        domain_filter = DomainFilter(
            allowed_domains=[domain],
            blocked_domains=[]
        )

        # Create DFS deep crawl strategy
        strategy = DFSDeepCrawlStrategy(
            max_depth=request.max_depth,
            include_external=False,  # Stay on same domain
            max_pages=request.max_pages,
            filter_chain=FilterChain([domain_filter])
        )

        # Configure crawler
        crawl_config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            stream=False,
            verbose=True
        )

        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            print("Starting crawl...", flush=True)
            results = await crawler.arun(
                url=request.start_url,
                config=crawl_config
            )

            # Handle results (could be list or single result)
            if isinstance(results, list):
                crawled_pages = results
            else:
                crawled_pages = [results] if results else []

            # Filter out failed results
            successful_pages = [r for r in crawled_pages if hasattr(r, 'success') and r.success]

            print(f"Crawled {len(successful_pages)} pages successfully", flush=True)

            for i, page in enumerate(successful_pages[:5], 1):
                url = page.url if hasattr(page, 'url') else 'unknown'
                print(f"  {i}. {url}", flush=True)

            if not successful_pages:
                return CrawlResponse(
                    success=False,
                    answer="No content could be crawled from the provided URL.",
                    pages_crawled=0,
                    sources=[],
                    message="Crawling failed - no pages found"
                )

            # Format context for DeepSeek
            context = format_context_for_llm(successful_pages)

            if not context.strip():
                return CrawlResponse(
                    success=False,
                    answer="Pages were crawled but no readable content was extracted.",
                    pages_crawled=len(successful_pages),
                    sources=[],
                    message="No content extracted"
                )

            print(f"\nContext length: {len(context)} chars", flush=True)
            print("Generating answer with DeepSeek-reasoner...", flush=True)

            answer = await call_deepseek(
                query=request.query,
                context=context,
                api_key=deepseek_api_key
            )

            # Prepare sources
            sources = []
            for page in successful_pages:
                url = page.url if hasattr(page, 'url') else str(page)
                sources.append({"url": url})

            return CrawlResponse(
                success=True,
                answer=answer,
                pages_crawled=len(successful_pages),
                sources=sources[:10],
                message=f"Crawled {len(successful_pages)} pages from {domain}"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}...", flush=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
