# RAG Engine API Documentation

This document provides comprehensive documentation for the RAG Engine REST API.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Common Response Format](#common-response-format)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [API Endpoints](#api-endpoints)
8. [WebSocket API](#websocket-api)
9. [SDK Examples](#sdk-examples)
10. [Troubleshooting](#troubleshooting)

## Overview

The RAG Engine API provides endpoints for:
- Document ingestion and indexing
- Query processing with various strategies
- System health and monitoring
- Configuration management
- Evaluation and metrics

### API Version
Current API version: `v1`

### Content Types
- Request: `application/json`
- Response: `application/json`

## Authentication

### API Key Authentication

Include your API key in the request headers:

```http
Authorization: Bearer YOUR_API_KEY
```

### Example
```bash
curl -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     https://api.yourdomain.com/api/v1/health
```

## Base URL

- **Production**: `https://api.yourdomain.com/api/v1`
- **Development**: `http://localhost:8000/api/v1`

## Common Response Format

All API responses follow this structure:

```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### Success Response
```json
{
  "success": true,
  "data": {
    "answer": "AI is a branch of computer science...",
    "confidence_score": 0.85,
    "processing_time": 1.23
  },
  "message": "Query processed successfully",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Query parameter is required",
    "details": {
      "field": "query",
      "reason": "missing_required_field"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Too Many Requests |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request validation failed |
| `UNAUTHORIZED` | Authentication required |
| `FORBIDDEN` | Insufficient permissions |
| `NOT_FOUND` | Resource not found |
| `RATE_LIMITED` | Rate limit exceeded |
| `PROCESSING_ERROR` | Query processing failed |
| `SYSTEM_ERROR` | Internal system error |

## Rate Limiting

- **Default**: 100 requests per minute per API key
- **Burst**: Up to 20 requests in 10 seconds

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## API Endpoints

### Health and Status

#### GET /health
Check system health status.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600,
    "components": {
      "database": "healthy",
      "cache": "healthy",
      "llm": "healthy"
    }
  }
}
```

#### GET /status
Get detailed system status and metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "system_info": {
      "version": "1.0.0",
      "environment": "production",
      "uptime": 3600
    },
    "stats": {
      "indexed_documents": 1000,
      "total_queries": 5000,
      "avg_response_time": 1.2
    },
    "health": {
      "status": "healthy",
      "checks": {
        "database": true,
        "cache": true,
        "llm": true
      }
    }
  }
}
```

### Document Management

#### POST /documents
Add documents to the knowledge base.

**Request:**
```json
{
  "documents": [
    {
      "content": "Document content here...",
      "metadata": {
        "title": "Document Title",
        "source": "web",
        "category": "technology"
      },
      "doc_id": "doc_001"
    }
  ],
  "indexing_strategy": "basic"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "indexed_count": 1,
    "processing_time": 2.5,
    "document_ids": ["doc_001"]
  },
  "message": "Documents indexed successfully"
}
```

#### GET /documents
List indexed documents with pagination.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20, max: 100)
- `filter` (string): Filter by metadata fields

**Response:**
```json
{
  "success": true,
  "data": {
    "documents": [
      {
        "doc_id": "doc_001",
        "title": "Document Title",
        "indexed_at": "2024-01-15T10:30:00Z",
        "metadata": {
          "source": "web",
          "category": "technology"
        }
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 100,
      "pages": 5
    }
  }
}
```

#### DELETE /documents/{doc_id}
Remove a document from the knowledge base.

**Response:**
```json
{
  "success": true,
  "data": {
    "doc_id": "doc_001",
    "removed": true
  },
  "message": "Document removed successfully"
}
```

#### DELETE /documents
Clear all documents from the knowledge base.

**Response:**
```json
{
  "success": true,
  "data": {
    "removed_count": 100
  },
  "message": "All documents cleared"
}
```

### Query Processing

#### POST /query
Process a query using the RAG system.

**Request:**
```json
{
  "query": "What is artificial intelligence?",
  "strategy": "multi_query",
  "k": 5,
  "metadata_filter": {
    "category": "technology"
  },
  "options": {
    "temperature": 0.1,
    "max_tokens": 500
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "answer": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence...",
    "confidence_score": 0.85,
    "processing_time": 1.23,
    "source_documents": [
      {
        "doc_id": "doc_001",
        "content": "Relevant document content...",
        "relevance_score": 0.92,
        "metadata": {
          "title": "AI Introduction",
          "source": "textbook"
        }
      }
    ],
    "metadata": {
      "strategy_used": "multi_query",
      "retrieved_count": 5,
      "model_used": "gemini-1.5-flash"
    }
  }
}
```

#### POST /query/batch
Process multiple queries in batch.

**Request:**
```json
{
  "queries": [
    {
      "id": "q1",
      "query": "What is AI?",
      "strategy": "basic"
    },
    {
      "id": "q2", 
      "query": "How does ML work?",
      "strategy": "multi_query"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "q1",
        "answer": "AI is...",
        "confidence_score": 0.85,
        "processing_time": 1.1
      },
      {
        "id": "q2",
        "answer": "ML works by...",
        "confidence_score": 0.78,
        "processing_time": 1.5
      }
    ],
    "total_processing_time": 2.6
  }
}
```

### Query Strategies

#### GET /strategies
List available query processing strategies.

**Response:**
```json
{
  "success": true,
  "data": {
    "strategies": [
      {
        "name": "basic",
        "description": "Standard RAG processing",
        "parameters": []
      },
      {
        "name": "multi_query",
        "description": "Generate multiple query variations",
        "parameters": [
          {
            "name": "num_queries",
            "type": "integer",
            "default": 3,
            "description": "Number of query variations to generate"
          }
        ]
      },
      {
        "name": "rag_fusion",
        "description": "RAG-Fusion with reciprocal rank fusion",
        "parameters": []
      }
    ]
  }
}
```

### Evaluation

#### POST /evaluate
Evaluate the RAG system with test cases.

**Request:**
```json
{
  "test_cases": [
    {
      "question": "What is AI?",
      "expected_answer": "AI is a branch of computer science...",
      "metadata": {
        "category": "definition"
      }
    }
  ],
  "metrics": ["custom", "ragas"],
  "options": {
    "detailed_results": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "overall_score": 0.82,
    "metric_scores": {
      "faithfulness": 0.85,
      "relevancy": 0.80,
      "correctness": 0.81
    },
    "test_case_results": [
      {
        "question": "What is AI?",
        "generated_answer": "AI is a branch...",
        "scores": {
          "faithfulness": 0.85,
          "relevancy": 0.80
        }
      }
    ],
    "recommendations": [
      "Consider improving document coverage for AI topics"
    ]
  }
}
```

### Configuration

#### GET /config
Get current system configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "llm_provider": "google",
    "llm_model": "gemini-1.5-flash",
    "embedding_provider": "google",
    "vector_store": "chroma",
    "indexing_strategy": "basic",
    "chunk_size": 1000,
    "retrieval_k": 5
  }
}
```

#### PUT /config
Update system configuration.

**Request:**
```json
{
  "chunk_size": 800,
  "retrieval_k": 7,
  "temperature": 0.2
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "updated_fields": ["chunk_size", "retrieval_k", "temperature"],
    "restart_required": false
  },
  "message": "Configuration updated successfully"
}
```

### Monitoring

#### GET /metrics
Get system metrics in Prometheus format.

**Response:**
```
# HELP rag_engine_requests_total Total number of requests
# TYPE rag_engine_requests_total counter
rag_engine_requests_total{method="POST",endpoint="/query",status="200"} 1000

# HELP rag_engine_request_duration_seconds Request duration in seconds
# TYPE rag_engine_request_duration_seconds histogram
rag_engine_request_duration_seconds_bucket{le="0.1"} 100
rag_engine_request_duration_seconds_bucket{le="0.5"} 800
rag_engine_request_duration_seconds_bucket{le="1.0"} 950
rag_engine_request_duration_seconds_bucket{le="+Inf"} 1000
```

#### GET /metrics/summary
Get human-readable metrics summary.

**Response:**
```json
{
  "success": true,
  "data": {
    "requests": {
      "total": 10000,
      "success_rate": 0.98,
      "avg_response_time": 1.2
    },
    "queries": {
      "total": 8000,
      "avg_confidence": 0.82,
      "avg_processing_time": 1.1
    },
    "system": {
      "uptime": 86400,
      "memory_usage": 0.65,
      "cpu_usage": 0.45
    }
  }
}
```

## WebSocket API

### Connection
Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Authentication
Send authentication message after connection:

```json
{
  "type": "auth",
  "token": "your-api-key"
}
```

### Query Streaming
Stream query processing results:

```json
{
  "type": "query",
  "data": {
    "query": "What is AI?",
    "stream": true
  }
}
```

**Response Stream:**
```json
{"type": "progress", "data": {"stage": "retrieval", "progress": 0.3}}
{"type": "progress", "data": {"stage": "generation", "progress": 0.7}}
{"type": "result", "data": {"answer": "AI is...", "confidence": 0.85}}
```

## SDK Examples

### Python SDK

```python
from rag_engine_sdk import RAGClient

# Initialize client
client = RAGClient(
    base_url="https://api.yourdomain.com/api/v1",
    api_key="your-api-key"
)

# Add documents
documents = [
    {
        "content": "AI is a branch of computer science...",
        "metadata": {"title": "AI Introduction", "source": "textbook"}
    }
]
result = client.add_documents(documents)
print(f"Indexed {result['indexed_count']} documents")

# Query
response = client.query(
    query="What is artificial intelligence?",
    strategy="multi_query",
    k=5
)
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence_score']}")

# Evaluate
test_cases = [
    {
        "question": "What is AI?",
        "expected_answer": "AI is a branch of computer science..."
    }
]
evaluation = client.evaluate(test_cases, metrics=["custom"])
print(f"Overall score: {evaluation['overall_score']}")
```

### JavaScript SDK

```javascript
import { RAGClient } from 'rag-engine-sdk';

// Initialize client
const client = new RAGClient({
  baseURL: 'https://api.yourdomain.com/api/v1',
  apiKey: 'your-api-key'
});

// Add documents
const documents = [
  {
    content: 'AI is a branch of computer science...',
    metadata: { title: 'AI Introduction', source: 'textbook' }
  }
];

const result = await client.addDocuments(documents);
console.log(`Indexed ${result.indexed_count} documents`);

// Query
const response = await client.query({
  query: 'What is artificial intelligence?',
  strategy: 'multi_query',
  k: 5
});

console.log(`Answer: ${response.answer}`);
console.log(`Confidence: ${response.confidence_score}`);

// Stream query
const stream = client.queryStream({
  query: 'What is AI?',
  onProgress: (progress) => console.log('Progress:', progress),
  onResult: (result) => console.log('Result:', result)
});
```

### cURL Examples

#### Add Documents
```bash
curl -X POST "https://api.yourdomain.com/api/v1/documents" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "AI is a branch of computer science...",
        "metadata": {"title": "AI Introduction"},
        "doc_id": "doc_001"
      }
    ]
  }'
```

#### Query
```bash
curl -X POST "https://api.yourdomain.com/api/v1/query" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "strategy": "multi_query",
    "k": 5
  }'
```

#### Health Check
```bash
curl -X GET "https://api.yourdomain.com/api/v1/health" \
  -H "Authorization: Bearer your-api-key"
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```json
   {
     "success": false,
     "error": {
       "code": "UNAUTHORIZED",
       "message": "Invalid API key"
     }
   }
   ```
   **Solution**: Verify your API key is correct and active.

2. **Rate Limiting**
   ```json
   {
     "success": false,
     "error": {
       "code": "RATE_LIMITED",
       "message": "Rate limit exceeded"
     }
   }
   ```
   **Solution**: Implement exponential backoff or reduce request frequency.

3. **Processing Errors**
   ```json
   {
     "success": false,
     "error": {
       "code": "PROCESSING_ERROR",
       "message": "Query processing failed",
       "details": {
         "reason": "no_documents_indexed"
       }
     }
   }
   ```
   **Solution**: Ensure documents are indexed before querying.

### Debug Headers

Include debug headers for troubleshooting:

```http
X-Debug-Mode: true
X-Trace-Request: true
```

### Support

- **Documentation**: https://docs.yourdomain.com
- **Status Page**: https://status.yourdomain.com
- **Support**: support@yourdomain.com

### API Changelog

#### v1.1.0 (2024-02-01)
- Added batch query processing
- Improved error messages
- Added WebSocket support

#### v1.0.0 (2024-01-15)
- Initial API release
- Basic RAG functionality
- Document management
- Query processing