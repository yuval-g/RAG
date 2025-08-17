# WebSocket API Reference

Documentation for the RAG System WebSocket API.

## Overview

The RAG System provides a WebSocket API for real-time interactions and streaming responses, enabling dynamic updates and efficient communication for long-running operations like query processing.

## Connection

To establish a WebSocket connection, use the following URL:

```
ws://localhost:8000/ws
```

Replace `localhost:8000` with the actual host and port of your RAG Engine API server.

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('WebSocket connection established.');
  // Send authentication message after connection is open
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-api-key' // Replace with your actual API key
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received message:', message);
  // Handle different message types (e.g., progress, result)
};

ws.onclose = () => {
  console.log('WebSocket connection closed.');
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

## Authentication

After establishing the WebSocket connection, you must send an authentication message with your API key. This should be the first message sent over the WebSocket.

**Authentication Message Format:**
```json
{
  "type": "auth",
  "token": "your-api-key"
}
```

Replace `your-api-key` with your valid RAG Engine API key.

## Message Formats

All messages sent and received over the WebSocket are in JSON format.

### Client-to-Server Messages

#### Query Streaming Request

To initiate a query and receive streaming updates, send a message with `type: "query"` and the query details in the `data` field. Set `stream: true` to enable streaming.

```json
{
  "type": "query",
  "data": {
    "query": "What is artificial intelligence?",
    "stream": true,
    "strategy": "multi_query",
    "k": 5
  }
}
```

**Fields in `data`:**
*   `query` (string, required): The user's question or prompt.
*   `stream` (boolean, required): Must be `true` to enable streaming responses.
*   `strategy` (string, optional): The query processing strategy to use (e.g., "basic", "multi_query", "rag_fusion"). Defaults to "basic".
*   `k` (integer, optional): The number of relevant documents to retrieve. Defaults to 5.
*   Other fields as supported by the `/query` REST endpoint (e.g., `metadata_filter`, `options`).

### Server-to-Client Messages

#### Progress Updates

During long-running operations like query processing, the server may send progress updates.

```json
{
  "type": "progress",
  "data": {
    "stage": "retrieval",
    "progress": 0.3,
    "message": "Retrieving documents..."
  }
}
```

**Fields in `data`:**
*   `stage` (string): The current stage of processing (e.g., "retrieval", "generation", "embedding").
*   `progress` (float): A value between 0.0 and 1.0 indicating the completion percentage of the current stage.
*   `message` (string, optional): A human-readable message describing the current progress.

#### Final Result

Once the operation is complete, the server sends the final result.

```json
{
  "type": "result",
  "data": {
    "answer": "Artificial Intelligence (AI) is a branch of computer science...",
    "confidence_score": 0.85,
    "processing_time": 1.23,
    "source_documents": [
      // ... (array of source document objects)
    ],
    "metadata": {
      "strategy_used": "multi_query"
    }
  }
}
```

**Fields in `data`:**
*   Contains the same fields as the successful response from the `/query` REST endpoint.

#### Error Messages

If an error occurs during WebSocket communication or processing, an error message will be sent.

```json
{
  "type": "error",
  "data": {
    "code": "INVALID_MESSAGE",
    "message": "Invalid message format",
    "details": "The 'type' field is missing."
  }
}
```

**Fields in `data`:**
*   `code` (string): A machine-readable error code.
*   `message` (string): A human-readable error description.
*   `details` (string, optional): More specific details about the error.

## Events

WebSocket clients should handle the following events:

*   `onopen`: Fired when the connection is successfully established.
*   `onmessage`: Fired when a message is received from the server. The message content should be parsed as JSON to determine its `type` and `data`.
*   `onclose`: Fired when the connection is closed.
*   `onerror`: Fired when a WebSocket error occurs.

By handling these messages and events, clients can build interactive and real-time applications on top of the RAG Engine WebSocket API.
