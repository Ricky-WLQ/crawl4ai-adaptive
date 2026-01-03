# Crawl4AI Adaptive Crawler with Embedding-Based Relevance Scoring

An intelligent web crawler that uses **embedding-based semantic similarity** for adaptive crawling and **DeepSeek** for AI-powered answer generation.

## 🌟 Key Features

### Embedding-Based Relevance Scoring
- Uses `paraphrase-multilingual-mpnet-base-v2` for semantic understanding
- **50+ language support** - works with any language
- Scores URLs and content by semantic similarity to your query
- More accurate than keyword matching for understanding intent

### Adaptive Crawling
- **BestFirstCrawlingStrategy**: Prioritizes most relevant pages
- Intelligent link following based on semantic relevance
- Configurable depth, page limits, and score thresholds

### DeepSeek Integration
- Uses `deepseek-chat` model for answer generation
- Processes crawled content and provides direct answers
- Content ranked by relevance before processing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Embedding Model (Multilingual)                 │
│         paraphrase-multilingual-mpnet-base-v2               │
│                    (50+ languages)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                    Query Embedding
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              BestFirstCrawlingStrategy                      │
│    ┌──────────────────────────────────────────────────┐    │
│    │      EmbeddingRelevanceScorer                    │    │
│    │  - Scores URLs by semantic similarity            │    │
│    │  - Prioritizes most relevant links               │    │
│    └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                    Crawled Pages
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Content Relevance Scoring (Embeddings)            │
│         Re-ranks pages by content similarity                │
└─────────────────────────────────────────────────────────────┘
                              │
                   Ranked Content
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DeepSeek Chat                            │
│           Generates direct answer from content              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      Plain Text Answer
```

## 🚀 Setup

### Environment Variables (Zeabur Dashboard)

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes | Your DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com/) |
| `PORT` | No | Server port (default: 8080) |

### Zeabur Deployment

1. Connect your GitHub repository (`crawl4ai-adaptive`) to Zeabur
2. Zeabur will automatically detect the Dockerfile
3. Add `DEEPSEEK_API_KEY` in Zeabur dashboard → Environment Variables
4. Deploy!

### First Request

After deployment, call the warmup endpoint to pre-load the embedding model:
```bash
curl -X POST https://your-app.zeabur.app/warmup
```

## 📡 API Endpoints

### `GET /`
Service status and available endpoints.

### `GET /health`
Health check - shows embedding model and DeepSeek status.

### `POST /warmup`
Pre-load the embedding model to reduce first-request latency.

### `POST /crawl`
Main adaptive crawling endpoint with embedding-based scoring.

**Request Body:**
```json
{
  "start_url": "https://example.com",
  "query": "What are the pricing plans?",
  "max_depth": 2,
  "max_pages": 15,
  "include_external": false,
  "score_threshold": 0.3,
  "use_keywords_fallback": false
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_url` | string | required | Starting URL to crawl |
| `query` | string | required | Question or topic to search for |
| `max_depth` | int | 2 | Crawl depth (1-5) |
| `max_pages` | int | 15 | Maximum pages to crawl (1-50) |
| `include_external` | bool | false | Follow external domain links |
| `score_threshold` | float | 0.3 | Minimum relevance score (0-1) |
| `use_keywords_fallback` | bool | false | Use keyword scoring instead of embeddings |

**Response:**
```json
{
  "success": true,
  "answer": "Based on the website, there are three pricing tiers...",
  "pages_crawled": 8,
  "relevant_pages": [
    {
      "url": "https://example.com/pricing",
      "title": "Pricing - Example",
      "relevance_score": 0.8721,
      "content_preview": "Our pricing plans include...",
      "depth": 1
    }
  ],
  "query": "What are the pricing plans?",
  "message": "Successfully crawled 8 pages using embedding-based scoring",
  "scoring_method": "embedding"
}
```

### `POST /crawl/simple`
Single-page crawl with embedding scoring and DeepSeek analysis.

**Query Parameters:**
- `url`: URL to crawl
- `query`: Question to answer

## 🔧 How It Works

### 1. Query Embedding
Your query is converted to a 768-dimensional vector using the multilingual embedding model.

### 2. URL Scoring (Pre-crawl)
- Each discovered URL is scored by semantic similarity
- The `EmbeddingRelevanceScorer` combines:
  - Link anchor text
  - URL path segments
  - Surrounding context
- `BestFirstCrawlingStrategy` visits highest-scoring URLs first

### 3. Content Scoring (Post-crawl)
- All crawled page content is embedded
- Cosine similarity calculated against query embedding
- Pages re-ranked by content relevance

### 4. Answer Generation
- Top relevant pages sent to DeepSeek
- DeepSeek synthesizes information into a direct answer

## 🌍 Multilingual Support

The embedding model supports 50+ languages:
- Arabic, Bulgarian, Catalan, Czech, Danish, German, Greek, English, Spanish, Estonian, Persian, Finnish, French, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Lithuanian, Latvian, Dutch, Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Swedish, Thai, Turkish, Ukrainian, Vietnamese, Chinese, and more!

**Example multilingual queries:**
- English: "What are the pricing plans?"
- Spanish: "¿Cuáles son los planes de precios?"
- Japanese: "価格プランは何ですか？"
- German: "Was sind die Preispläne?"

## 📊 Scoring Methods

### Embedding-Based (Default)
- **Pros**: Understands semantic meaning, handles synonyms, multilingual
- **Cons**: Slightly slower, requires model loading
- Uses: `paraphrase-multilingual-mpnet-base-v2`

### Keyword-Based (Fallback)
- **Pros**: Fast, no model required
- **Cons**: Exact matching only, misses synonyms
- Enable with: `"use_keywords_fallback": true`

## 🐳 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run crawl4ai setup
crawl4ai-setup

# Set environment variable
export DEEPSEEK_API_KEY=your_api_key_here

# Run the server
python main.py
```

Access the API docs at `http://localhost:8080/docs`

## 📝 Example Usage

### Python
```python
import requests

response = requests.post(
    "https://your-app.zeabur.app/crawl",
    json={
        "start_url": "https://docs.example.com",
        "query": "How do I authenticate API requests?",
        "max_depth": 2,
        "max_pages": 20
    }
)

result = response.json()
print(result["answer"])
```

### cURL
```bash
curl -X POST "https://your-app.zeabur.app/crawl" \
  -H "Content-Type: application/json" \
  -d '{
    "start_url": "https://docs.example.com",
    "query": "How do I authenticate API requests?",
    "max_depth": 2,
    "max_pages": 20
  }'
```

## ⚙️ Performance Notes

- **First request**: ~30-60s (model loading)
- **Subsequent requests**: ~5-30s (depending on pages crawled)
- **Warmup endpoint**: Call `/warmup` after deployment to pre-load model
- **Memory**: ~2GB for embedding model + browser

## 📄 License

MIT
