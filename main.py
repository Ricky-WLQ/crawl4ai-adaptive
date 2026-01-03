"""
Crawl4AI Production Crawler - TRUE Adaptive Edition
====================================================
Features:
- Native AdaptiveCrawler with automatic stopping
- Embedding strategy with semantic understanding
- Three-layer scoring (Coverage, Consistency, Saturation)
- Confidence-based termination
- Knowledge base export/import
- DeepSeek AI answer generation
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import httpx
import traceback
import asyncio
import json
import uuid
import logging
from contextlib import asynccontextmanager

# ============== CONFIGURATION ==============

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PASSWORD = os.getenv("PASSWORD", "changeme")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_bases")

# Ensure knowledge base directory exists
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

# ============== JOB STORAGE ==============

class JobStorage:
    """Job storage with Redis or in-memory backend"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_store: Dict[str, dict] = {}
        self.job_ttl = 3600
        
    async def initialize(self):
        if REDIS_URL:
            try:
                import redis.asyncio as redis
                self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
                await self.redis_client.ping()
                logger.info("✅ Connected to Redis")
            except Exception as e:
                logger.warning(f"⚠️ Redis unavailable: {e}")
    
    async def set(self, job_id: str, data: dict):
        if self.redis_client:
            await self.redis_client.setex(f"job:{job_id}", self.job_ttl, json.dumps(data, default=str))
        else:
            self.memory_store[job_id] = data
    
    async def get(self, job_id: str) -> Optional[dict]:
        if self.redis_client:
            data = await self.redis_client.get(f"job:{job_id}")
            return json.loads(data) if data else None
        return self.memory_store.get(job_id)
    
    async def update(self, job_id: str, updates: dict):
        data = await self.get(job_id)
        if data:
            data.update(updates)
            await self.set(job_id, data)

job_storage = JobStorage()

# ============== AI ENGINE ==============

class AIEngine:
    """AI engine with DeepSeek integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    async def _call_api(self, messages: List[dict], model: str = "deepseek-chat",
                       max_tokens: int = 2000, timeout: int = 60) -> Optional[str]:
        if not self.api_key:
            return None
        
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        self.base_url,
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.1}
                    )
                    if response.status_code == 200:
                        return response.json()["choices"][0]["message"]["content"]
                    elif response.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"API error: {e}")
        return None
    
    async def generate_answer(self, query: str, context: str, 
                             use_reasoner: bool = True) -> dict:
        """Generate answer from retrieved context"""
        
        model = "deepseek-reasoner" if use_reasoner else "deepseek-chat"
        
        prompt = f"""Answer the query using ONLY the provided context. Be accurate and cite sources.

**QUERY:** {query}

**RETRIEVED CONTEXT:**
{context[:50000]}

**INSTRUCTIONS:**
1. Answer based ONLY on the provided context
2. Cite the source URL when making claims
3. If the context doesn't contain the answer, say so
4. Be precise and accurate
5. Include relevant code/examples if present in context"""

        messages = [
            {"role": "system", "content": "You are a precise research assistant. Only use provided context."},
            {"role": "user", "content": prompt}
        ]
        
        answer = await self._call_api(messages, model=model, max_tokens=3000, timeout=120)
        return {"answer": answer or "Unable to generate answer", "model_used": model}

# ============== ADAPTIVE CRAWLER SERVICE ==============

class AdaptiveCrawlerService:
    """
    Production-ready Adaptive Crawler using Crawl4AI's native implementation.
    
    Key Features:
    - Automatic stopping when sufficient information is gathered
    - Three-layer scoring: Coverage, Consistency, Saturation
    - Embedding strategy for semantic understanding
    - Knowledge base persistence
    """
    
    def __init__(self, config: dict, job_id: str = None):
        self.config = self._parse_config(config)
        self.job_id = job_id
        self.ai_engine = AIEngine(DEEPSEEK_API_KEY)
        
    def _parse_config(self, config: dict) -> dict:
        """Parse and validate configuration with sensible defaults"""
        defaults = {
            # Query
            "query": "",
            
            # Adaptive Crawling Settings
            "strategy": "embedding",  # "statistical" or "embedding"
            "confidence_threshold": 0.7,  # Stop when this confidence is reached
            "max_pages": 30,  # Maximum pages to crawl
            "max_depth": 5,  # Maximum crawl depth
            "top_k_links": 5,  # Links to follow per page
            "min_gain_threshold": 0.05,  # Minimum info gain to continue
            
            # Embedding Strategy Settings (when strategy="embedding")
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "n_query_variations": 10,  # Query variations for coverage
            "embedding_min_confidence_threshold": 0.1,  # Stop if completely irrelevant
            
            # State Persistence
            "save_state": True,
            "state_path": None,  # Auto-generated if None
            
            # Answer Generation
            "use_ai": True,
            "use_reasoner": True,
            "top_k_results": 10,  # Top pages for answer generation
            
            # Export
            "export_knowledge_base": True,
        }
        defaults.update(config)
        return defaults
    
    async def update_job_status(self, status: str, **kwargs):
        """Update job status in storage"""
        if self.job_id:
            await job_storage.update(self.job_id, {"status": status, **kwargs})
    
    async def crawl(self, start_urls: List[str]) -> dict:
        """
        Execute adaptive crawl with native Crawl4AI AdaptiveCrawler.
        
        The AdaptiveCrawler uses three metrics to measure information sufficiency:
        - Coverage: How well collected pages cover the query terms
        - Consistency: Whether information is coherent across pages
        - Saturation: Detecting when new pages aren't adding new information
        """
        
        try:
            from crawl4ai import AsyncWebCrawler, AdaptiveCrawler, AdaptiveConfig, BrowserConfig
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"crawl4ai not installed: {e}")
        
        start_time = datetime.now()
        query = self.config["query"]
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required for adaptive crawling")
        
        # Generate state path if not provided
        state_path = self.config["state_path"]
        if state_path is None and self.config["save_state"]:
            state_path = os.path.join(
                KNOWLEDGE_BASE_DIR, 
                f"crawl_state_{self.job_id or uuid.uuid4().hex[:8]}.json"
            )
        
        # Build AdaptiveConfig based on strategy
        await self.update_job_status("configuring", strategy=self.config["strategy"])
        
        if self.config["strategy"] == "embedding":
            # Embedding strategy - semantic understanding
            adaptive_config = AdaptiveConfig(
                strategy="embedding",
                embedding_model=self.config["embedding_model"],
                n_query_variations=self.config["n_query_variations"],
                embedding_min_confidence_threshold=self.config["embedding_min_confidence_threshold"],
                confidence_threshold=self.config["confidence_threshold"],
                max_pages=self.config["max_pages"],
                max_depth=self.config["max_depth"],
                top_k_links=self.config["top_k_links"],
                min_gain_threshold=self.config["min_gain_threshold"],
                save_state=self.config["save_state"],
                state_path=state_path
            )
            logger.info(f"📊 Using EMBEDDING strategy with {self.config['embedding_model']}")
        else:
            # Statistical strategy - term-based analysis
            adaptive_config = AdaptiveConfig(
                strategy="statistical",
                confidence_threshold=self.config["confidence_threshold"],
                max_pages=self.config["max_pages"],
                max_depth=self.config["max_depth"],
                top_k_links=self.config["top_k_links"],
                min_gain_threshold=self.config["min_gain_threshold"],
                save_state=self.config["save_state"],
                state_path=state_path
            )
            logger.info("📈 Using STATISTICAL strategy")
        
        # Execute Adaptive Crawl
        await self.update_job_status("crawling", query=query)
        
        all_results = []
        total_pages_crawled = 0
        final_confidence = 0.0
        coverage_stats = {}
        knowledge_base_size = 0
        
        browser_config = BrowserConfig(headless=True, verbose=False)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            adaptive = AdaptiveCrawler(crawler, adaptive_config)
            
            for start_url in start_urls:
                logger.info(f"🚀 Starting adaptive crawl from: {start_url}")
                logger.info(f"🔍 Query: {query}")
                
                try:
                    # The magic happens here - digest() handles everything
                    result = await adaptive.digest(
                        start_url=start_url,
                        query=query
                    )
                    
                    # Collect statistics
                    total_pages_crawled += len(result.crawled_urls)
                    final_confidence = adaptive.confidence
                    coverage_stats = adaptive.coverage_stats
                    
                    if hasattr(adaptive, 'state') and hasattr(adaptive.state, 'knowledge_base'):
                        knowledge_base_size = len(adaptive.state.knowledge_base)
                    
                    # Update job status with progress
                    await self.update_job_status(
                        "crawling",
                        pages_crawled=total_pages_crawled,
                        confidence=f"{final_confidence:.0%}",
                        current_url=start_url
                    )
                    
                    # Print stats to logs
                    adaptive.print_stats()
                    
                except Exception as e:
                    logger.error(f"Error crawling {start_url}: {e}")
                    continue
            
            # Get most relevant content
            await self.update_job_status("retrieving_content")
            relevant_pages = adaptive.get_relevant_content(top_k=self.config["top_k_results"])
            
            # Export knowledge base if configured
            kb_export_path = None
            if self.config["export_knowledge_base"]:
                kb_export_path = os.path.join(
                    KNOWLEDGE_BASE_DIR,
                    f"knowledge_base_{self.job_id or uuid.uuid4().hex[:8]}.jsonl"
                )
                try:
                    adaptive.export_knowledge_base(kb_export_path)
                    logger.info(f"📚 Knowledge base exported to: {kb_export_path}")
                except Exception as e:
                    logger.warning(f"Could not export knowledge base: {e}")
                    kb_export_path = None
        
        # Generate AI Answer
        ai_result = {"answer": "", "model_used": ""}
        if self.config["use_ai"] and relevant_pages:
            await self.update_job_status("generating_answer")
            
            # Format context from relevant pages
            context_parts = []
            for i, page in enumerate(relevant_pages[:15]):
                context_parts.append(
                    f"**Source {i+1}:** {page.get('url', 'Unknown')}\n"
                    f"**Relevance Score:** {page.get('score', 0):.2%}\n\n"
                    f"{page.get('content', '')[:3000]}"
                )
            
            context = "\n\n---\n\n".join(context_parts)
            
            ai_result = await self.ai_engine.generate_answer(
                query,
                context,
                self.config["use_reasoner"]
            )
        
        # Compile Final Results
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = {
            "success": True,
            "query": query,
            
            # Adaptive Crawling Metrics
            "adaptive_metrics": {
                "strategy": self.config["strategy"],
                "confidence_achieved": f"{final_confidence:.0%}",
                "confidence_threshold": f"{self.config['confidence_threshold']:.0%}",
                "coverage": coverage_stats.get("coverage", 0),
                "consistency": coverage_stats.get("consistency", 0),
                "saturation": coverage_stats.get("saturation", 0),
                "pages_crawled": total_pages_crawled,
                "max_pages_allowed": self.config["max_pages"],
                "knowledge_base_documents": knowledge_base_size,
                "efficiency": f"{(len(relevant_pages) / max(total_pages_crawled, 1)) * 100:.1f}% useful pages"
            },
            
            # AI Answer
            "answer": ai_result.get("answer", ""),
            "answer_metadata": {
                "model_used": ai_result.get("model_used", ""),
                "sources_used": len(relevant_pages)
            },
            
            # Retrieved Content
            "relevant_pages": [
                {
                    "url": page.get("url", ""),
                    "score": round(page.get("score", 0), 4),
                    "content_preview": (page.get("content", "") or "")[:500] + "..."
                        if len(page.get("content", "") or "") > 500 
                        else page.get("content", "")
                }
                for page in relevant_pages
            ],
            
            # Statistics
            "statistics": {
                "elapsed_seconds": round(elapsed, 2),
                "start_urls": start_urls,
                "embedding_model": self.config["embedding_model"] if self.config["strategy"] == "embedding" else None
            },
            
            # Export Paths
            "exports": {
                "knowledge_base": kb_export_path,
                "state_file": state_path if self.config["save_state"] else None
            }
        }
        
        await self.update_job_status(
            "completed", 
            result=result, 
            completed_at=datetime.now().isoformat()
        )
        
        return result

# ============== LIFESPAN ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    await job_storage.initialize()
    logger.info("🚀 Crawl4AI Adaptive Crawler Service Started")
    yield

app = FastAPI(
    title="Crawl4AI Adaptive Crawler",
    description="Production-ready crawler with native Crawl4AI adaptive intelligence",
    version="4.0.0",
    lifespan=lifespan
)

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

# ============== API MODELS ==============

class AdaptiveCrawlRequest(BaseModel):
    """Request model for adaptive crawling"""
    urls: List[str] = Field(..., description="Starting URLs for adaptive crawl")
    query: str = Field(..., description="Query to guide the adaptive crawl")
    
    # Strategy
    strategy: str = Field(
        default="embedding",
        description="Crawling strategy: 'statistical' (fast, exact terms) or 'embedding' (semantic understanding)"
    )
    
    # Adaptive Parameters
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Stop when this confidence is reached")
    max_pages: int = Field(default=30, ge=1, le=200, description="Maximum pages to crawl")
    max_depth: int = Field(default=5, ge=1, le=10, description="Maximum crawl depth")
    top_k_links: int = Field(default=5, ge=1, le=20, description="Links to follow per page")
    min_gain_threshold: float = Field(default=0.05, ge=0.0, le=0.5, description="Minimum info gain to continue")
    
    # Embedding Settings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    n_query_variations: int = Field(default=10, ge=1, le=50, description="Query variations for embedding strategy")
    
    # AI Settings
    use_ai: bool = Field(default=True, description="Generate AI answer from results")
    use_reasoner: bool = Field(default=True, description="Use DeepSeek Reasoner for answers")
    top_k_results: int = Field(default=10, ge=1, le=50, description="Top pages to use for answer")

class ResumeRequest(BaseModel):
    """Resume a previous crawl from saved state"""
    state_path: str = Field(..., description="Path to saved state file")
    query: str = Field(..., description="Query for the resumed crawl")
    start_url: str = Field(..., description="Starting URL")

# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    return {
        "service": "Crawl4AI Adaptive Crawler",
        "version": "4.0.0",
        "features": [
            "Native AdaptiveCrawler integration",
            "Automatic stopping when sufficient info gathered",
            "Three-layer scoring (Coverage, Consistency, Saturation)",
            "Statistical strategy (fast, exact terms)",
            "Embedding strategy (semantic understanding)",
            "Knowledge base export/import",
            "State persistence and resume",
            "AI-powered answer generation"
        ],
        "strategies": {
            "statistical": "Uses pure information theory and term-based analysis. Fast and efficient with no API calls.",
            "embedding": "Uses sentence-transformers for semantic understanding. Automatically expands query into variations."
        },
        "endpoints": {
            "/crawl": "Synchronous adaptive crawl",
            "/crawl/async": "Asynchronous adaptive crawl with job tracking",
            "/job/{job_id}": "Get job status and results",
            "/resume": "Resume crawl from saved state"
        }
    }

@app.get("/health")
async def health():
    try:
        from crawl4ai import AdaptiveCrawler, AdaptiveConfig
        crawl4ai_available = True
    except ImportError:
        crawl4ai_available = False
    
    return {
        "status": "healthy",
        "crawl4ai_available": crawl4ai_available,
        "deepseek_configured": bool(DEEPSEEK_API_KEY)
    }

@app.post("/crawl")
async def crawl_sync(request: AdaptiveCrawlRequest, credentials = Depends(verify_token)):
    """
    Synchronous adaptive crawl.
    
    The crawler will automatically:
    - Follow only relevant links based on the query
    - Stop when sufficient information is gathered (confidence threshold)
    - Track coverage, consistency, and saturation metrics
    - Generate an AI-powered answer from the results
    """
    config = request.model_dump()
    service = AdaptiveCrawlerService(config)
    
    try:
        return await service.crawl(request.urls)
    except Exception as e:
        logger.error(f"Crawl error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl/async")
async def crawl_async(request: AdaptiveCrawlRequest, background_tasks: BackgroundTasks,
                     credentials = Depends(verify_token)):
    """
    Asynchronous adaptive crawl with job tracking.
    
    Returns immediately with a job_id. Use /job/{job_id} to check status.
    """
    job_id = str(uuid.uuid4())
    config = request.model_dump()
    
    await job_storage.set(job_id, {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "query": request.query,
        "urls": request.urls,
        "strategy": request.strategy
    })
    
    async def run_crawl():
        try:
            service = AdaptiveCrawlerService(config, job_id)
            await service.crawl(request.urls)
        except Exception as e:
            await job_storage.update(job_id, {
                "status": "failed", 
                "error": str(e),
                "traceback": traceback.format_exc()
            })
    
    background_tasks.add_task(run_crawl)
    return {"job_id": job_id, "status": "queued"}

@app.get("/job/{job_id}")
async def get_job(job_id: str, credentials = Depends(verify_token)):
    """Get job status and results"""
    job = await job_storage.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/resume")
async def resume_crawl(request: ResumeRequest, credentials = Depends(verify_token)):
    """
    Resume a previous crawl from saved state.
    
    Useful for:
    - Continuing interrupted crawls
    - Extending existing knowledge bases
    - Adding new queries to existing crawl data
    """
    try:
        from crawl4ai import AsyncWebCrawler, AdaptiveCrawler, AdaptiveConfig, BrowserConfig
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"crawl4ai not installed: {e}")
    
    if not os.path.exists(request.state_path):
        raise HTTPException(status_code=404, detail=f"State file not found: {request.state_path}")
    
    config = AdaptiveConfig(
        save_state=True,
        state_path=request.state_path
    )
    
    browser_config = BrowserConfig(headless=True, verbose=False)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        adaptive = AdaptiveCrawler(crawler, config)
        
        # Resume from saved state
        result = await adaptive.digest(
            start_url=request.start_url,
            query=request.query,
            resume_from=request.state_path
        )
        
        adaptive.print_stats()
        relevant_pages = adaptive.get_relevant_content(top_k=10)
        
        return {
            "success": True,
            "resumed_from": request.state_path,
            "pages_crawled": len(result.crawled_urls),
            "confidence": f"{adaptive.confidence:.0%}",
            "relevant_pages": [
                {
                    "url": p.get("url", ""),
                    "score": round(p.get("score", 0), 4)
                }
                for p in relevant_pages
            ]
        }

@app.post("/import-knowledge-base")
async def import_knowledge_base(path: str, credentials = Depends(verify_token)):
    """Import a previously exported knowledge base"""
    try:
        from crawl4ai import AsyncWebCrawler, AdaptiveCrawler, BrowserConfig
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"crawl4ai not installed: {e}")
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Knowledge base not found: {path}")
    
    browser_config = BrowserConfig(headless=True, verbose=False)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        adaptive = AdaptiveCrawler(crawler)
        adaptive.import_knowledge_base(path)
        
        return {
            "success": True,
            "imported_from": path,
            "message": "Knowledge base imported successfully"
        }

@app.get("/strategies")
async def list_strategies():
    """Compare available crawling strategies"""
    return {
        "statistical": {
            "description": "Uses pure information theory and term-based analysis",
            "pros": ["Fast and efficient", "No API calls or model loading", "Good for exact term matching"],
            "cons": ["Literal matching only", "May miss semantic variations"],
            "best_for": ["Technical documentation", "API references", "Exact term searches"],
            "example_config": {
                "strategy": "statistical",
                "confidence_threshold": 0.7,
                "max_pages": 20
            }
        },
        "embedding": {
            "description": "Uses sentence-transformers for semantic understanding",
            "pros": ["Understands concepts and synonyms", "Automatic query expansion", "Higher precision"],
            "cons": ["Requires model loading", "Slightly slower initial startup"],
            "best_for": ["Research queries", "Conceptual searches", "When exact terms unknown"],
            "example_config": {
                "strategy": "embedding",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "n_query_variations": 10,
                "confidence_threshold": 0.8
            }
        },
        "comparison_example": {
            "query": "authentication oauth",
            "statistical_result": "Searches for exact terms, 12 pages, 78% confidence",
            "embedding_result": "Understands 'auth', 'login', 'SSO' - 8 pages, 92% confidence"
        }
    }

# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║          Crawl4AI TRUE Adaptive Crawler v4.0.0                  ║
╠══════════════════════════════════════════════════════════════════╣
║  🧠 Native AdaptiveCrawler Integration                          ║
║  📊 Three-Layer Scoring: Coverage, Consistency, Saturation      ║
║  🎯 Automatic Stopping at Confidence Threshold                  ║
║  🔍 Two Strategies: Statistical & Embedding                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Strategies:                                                     ║
║  • statistical: Fast, term-based, no model loading              ║
║  • embedding: Semantic understanding, query expansion           ║
╠══════════════════════════════════════════════════════════════════╣
║  Key Difference from Traditional Crawling:                       ║
║  Traditional: 500 pages → 50 useful → $15 tokens → 2 hours      ║
║  Adaptive:     15 pages → 14 useful →  $2 tokens → 10 minutes   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run("main:app", host="0.0.0.0", port=port)
