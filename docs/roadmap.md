# Roadmap

This roadmap outlines the long-term vision for Remem. The project is developed incrementally, with each release building toward a production-grade AI infrastructure library.

## Release Overview

| Version | Focus | Status |
|---|---|---|
| v1.0.0-beta | Local-first AI work reuse engine for early adopters | Completed |
| v1.0.0 | Stable local-first AI work reuse engine | Completed |
| v1.1.0 | Production-ready optional ANN search | Completed |
| v1.2.0 | Redis-backed distributed semantic cache | **Current release** |
| v1.3.0 | Distributed scale and additional storage backends | Planned |
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
- Optional HNSW candidate retrieval with exact cosine reranking
- Direct candidate-ID storage lookup without query-time storage scans
- Incremental and persistent namespace-partitioned ANN indexes
- Validated fast reload, consistency recovery, and structured-policy filtering
- Python 3.10, 3.11, and 3.12 CI coverage with lint, type, and package checks
- Optional Redis shared storage with local fallback and reconnect replay
- Cross-node response and retrieval reuse through the existing policy engine
- Best-effort duplicate-work coalescing with expiring token locks
- Distributed metrics and real Redis integration coverage

**Planned**

*Performance*
- Arbitrary metadata indexing after explicit policy semantics are designed
- Further retrieval and memory optimizations

*Storage*
- SQLite backend
- PostgreSQL backend
- Remote vector candidate discovery and distributed ANN alternatives
- Redis Cluster and larger-cache pagination/partitioning

*Developer experience*
- Improved configuration ergonomics
- CLI support
- Richer examples and documentation
- Documentation link checking in CI

## Version 2.x — Production Deployments

- Multi-region cache coordination
- Durable fallback queues
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
