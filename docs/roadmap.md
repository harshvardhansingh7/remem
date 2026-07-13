# Roadmap

This roadmap outlines the long-term vision for Remem. The project is developed incrementally, with each release building toward a production-grade AI infrastructure library.

## Release Overview

| Version | Focus | Status |
|---|---|---|
| v1.0.0-beta | Local-first AI work reuse engine for early adopters | Completed |
| v1.0.0 | Stable local-first AI work reuse engine | **Current release** |
| v1.1.0 | Approximate Nearest Neighbor (ANN) search | **In development** |
| v1.2.0 | Distributed semantic cache | Planned |
| v1.3.0 | Additional storage backends (Redis, Postgres, S3) | Planned |
| v1.4.0 | Advanced policy engine with ML-based thresholds | Planned |
| v2.0.0 | Multi-tenant support and cloud-native deployment | Planned |

## Version 1.x

**Completed**

- Semantic similarity engine
- Reuse engine (full / partial / miss decisions)
- Metadata policy engine
- Durable, atomic persistent storage
- Built-in observability and metrics
- Python SDK
- PyPI distribution
- Unit tests for persistence, similarity, and policy behavior

**Planned**

*Production hardening*
- Broader unit and integration test coverage
- Clearer release validation checklist
- Documented production deployment guidance
- Python 3.10 coverage in CI

*Performance*
- Optional HNSW cosine search with rebuildable in-memory indexes (`1.1.0.dev0`, completed)
- User-oriented automatic and exact/HNSW search modes (`1.1.0.dev1`, completed)
- Exact cosine reranking of ANN candidates (`1.1.0.dev2`, completed)
- Record-ID lookup without query-time full storage scans
- Incremental and persistent ANN indexes with consistency recovery
- Namespace-aware filtering; arbitrary metadata indexing requires further design
- Faster retrieval and memory optimizations

*Storage*
- SQLite backend
- PostgreSQL backend
- Redis backend

*Developer experience*
- Improved configuration ergonomics
- CLI support
- Richer examples and documentation
- Documentation link checking in CI

## Version 2.x — Production Deployments

- Distributed cache
- Remote storage
- Multi-process synchronization
- Async API
- Streaming support
- Horizontal scalability

## Version 3.x — AI Infrastructure Platform

- Cloud deployment
- Enterprise features
- Dashboard and observability UI
- Formal benchmark suite
- Advanced policy engine
- Plugin ecosystem

## Research Directions

Longer-term research interests that may shape future versions:

- Intelligent execution reuse beyond request/response caching
- Adaptive semantic caching (self-tuning thresholds)
- Agent memory optimization
- Cost-aware LLM routing
- Retrieval optimization strategies

---

Roadmaps evolve over time. Priorities may change based on community feedback and research findings. Have a feature request? Open an issue — see [CONTRIBUTING.md](../CONTRIBUTING.md).
