# API Reference

This directory contains detailed documentation for all RAG Engine APIs, including command-line interface, REST API, WebSocket API, and Python SDK.

## Table of Contents

- [CLI Reference](./cli-reference.md) - Command-line interface documentation
- [REST API](./rest-api.md) - HTTP REST API endpoints and usage
- [WebSocket API](./websocket-api.md) - WebSocket interface for real-time communication
- [Python SDK](./python-sdk.md) - Python software development kit

## Getting Started

If you're new to using the RAG Engine APIs:

1. For local usage and scripting, start with the [CLI Reference](./cli-reference.md)
2. For integrating with web applications, use the [REST API](./rest-api.md)
3. For real-time applications, consider the [WebSocket API](./websocket-api.md)
4. For Python applications, the [Python SDK](./python-sdk.md) provides the most convenient interface

## API Comparison

| API Type | Best For | Complexity | Real-time | Authentication |
|----------|----------|------------|-----------|----------------|
| [CLI](./cli-reference.md) | Local usage, scripts | Low | No | Simple |
| [REST](./rest-api.md) | Web applications | Medium | No | API Keys |
| [WebSocket](./websocket-api.md) | Real-time applications | Medium | Yes | API Keys |
| [Python SDK](./python-sdk.md) | Python applications | Low | Yes | API Keys |

## Authentication

All APIs (except CLI) require authentication:

- **API Keys**: Most common method for REST, WebSocket, and Python SDK
- **JWT**: Advanced authentication for enterprise deployments
- **OAuth**: For integration with existing identity providers

For authentication setup, see the [Security Configuration](../configuration/security.md) guide.

## Rate Limiting

API rate limiting is configurable and enforced:

- Default: 100 requests per minute
- Burst: Up to 20 requests in 10 seconds
- Custom limits can be configured in [Configuration](../configuration/configuration-reference.md)

## Error Handling

All APIs follow consistent error handling patterns:

- HTTP status codes for REST API
- Standard error objects for all interfaces
- Detailed error messages for troubleshooting

For error handling implementation details, see [Error Handling and Resilience](../development/error_handling_and_resilience.md).

## Versioning

APIs follow semantic versioning:

- Major versions (v1, v2) for breaking changes
- Minor versions (v1.1, v1.2) for new features
- Patch versions (v1.0.1, v1.0.2) for bug fixes

Current version: v1

## Related Documentation

- [Configuration Reference](../configuration/configuration-reference.md) - API configuration options
- [Security Configuration](../configuration/security.md) - Authentication and authorization setup
- [Deployment Guides](../deployment/) - How to deploy API services
- [Monitoring](../operations/monitoring.md) - API performance monitoring