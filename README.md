# Enterprise Data Assistant

### AI-Powered Enterprise Analytics & Intelligent Document Retrieval

An AI-powered enterprise data assistant that combines **Large Language Models (LLMs), Natural Language-to-SQL (NL2SQL), Retrieval-Augmented Generation (RAG), and semantic search** to enable intelligent data access and context-aware question answering.

The platform integrates structured enterprise data querying with document-based knowledge retrieval, helping users interact with business data and technical documentation through natural language.

---

## Overview

Enterprise Data Assistant brings together two AI-powered capabilities in a unified platform:

- **CRI — Conversational Enterprise Data Assistant:** Converts natural language questions into SQL queries to retrieve relevant information from enterprise databases.
- **DocMind — Intelligent Document Assistant:** Uses Retrieval-Augmented Generation (RAG) and semantic retrieval to answer questions using relevant documents and technical knowledge.

The system combines AI inference, backend APIs, database connectivity, metadata-driven retrieval, and containerized deployment to support enterprise knowledge access and analytics.

## Key Features

### 1. Natural Language-to-SQL

- Converts natural language questions into SQL queries.
- Enables conversational access to structured enterprise data.
- Uses schema and metadata context to support SQL generation.
- Supports graph-based join generation using database relationship metadata.
- Incorporates SQL validation to improve query reliability.
- Supports module-aware query processing for enterprise workflows.

### 2. Retrieval-Augmented Generation (RAG)

- Retrieves relevant information from a knowledge base to ground AI-generated answers.
- Uses semantic search to identify contextually relevant content.
- Supports document-based question answering.
- Uses metadata-aware retrieval to improve relevance.
- Supports module-based knowledge access.
- Helps users find information from technical documentation and machine manuals.

### 3. Intelligent Document Retrieval

- Supports document ingestion and knowledge-base workflows.
- Uses vector search for semantic retrieval.
- Supports machine-aware and metadata-based document filtering.
- Connects retrieved context with LLM-based answer generation.
- Designed for technical documentation and enterprise knowledge use cases.

### 4. AI Model Integration

- Integrates Large Language Models into enterprise question-answering workflows.
- Supports local LLM inference through the Ollama ecosystem.
- Separates AI inference from application and API layers.
- Supports configurable model selection and inference settings.

### 5. Backend APIs

- Uses FastAPI to expose backend functionality.
- Provides API-based communication between application components.
- Integrates database access, retrieval, and AI-generated responses.
- Supports containerized application deployment.

### 6. Containerized Deployment

- Uses Docker to package application components.
- Separates application services for easier deployment and maintenance.
- Supports configurable service ports and environment settings.
- Helps maintain consistent application environments.

---

## System Architecture

The platform consists of the following logical components:

```text
                    ┌─────────────────────────────┐
                    │          End Users          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Application UI         │
                    └──────────────┬──────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   │                               │
                   ▼                               ▼
       ┌──────────────────────┐       ┌──────────────────────┐
       │         CRI          │       │       DocMind        │
       │ Enterprise Data      │       │ Document Knowledge   │
       │ Assistant            │       │ Assistant            │
       └──────────┬───────────┘       └──────────┬───────────┘
                  │                              │
                  ▼                              ▼
       ┌──────────────────────┐       ┌──────────────────────┐
       │ Natural Language     │       │ Document Retrieval   │
       │ to SQL Pipeline      │       │ and Semantic Search  │
       └──────────┬───────────┘       └──────────┬───────────┘
                  │                              │
                  ▼                              ▼
       ┌──────────────────────┐       ┌──────────────────────┐
       │ Schema & Join        │       │ Vector Database      │
       │ Metadata             │       │ Qdrant               │
       └──────────┬───────────┘       └──────────┬───────────┘
                  │                              │
                  ▼                              ▼
       ┌──────────────────────┐       ┌──────────────────────┐
       │ Enterprise Database  │       │ Retrieved Document   │
       │ SQL / PostgreSQL     │       │ Context              │
       └──────────────────────┘       └──────────┬───────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │ LLM-Based Answer       │
                                    │ Generation             │
                                    └────────────────────────┘
```

*Logical architecture illustration. Actual integrations and data flows depend on the configured deployment.*

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| AI & Language Models | Large Language Models (LLMs), Ollama |
| Natural Language Processing | Natural Language-to-SQL, contextual question answering |
| Retrieval Architecture | Retrieval-Augmented Generation (RAG) |
| Semantic Search | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Database | Qdrant |
| Graph-Based Relationships | Neo4j |
| Relational Databases | SQL Server, PostgreSQL |
| Caching | Redis-compatible caching infrastructure |
| Containerization | Docker |
| API Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |

*The active technology integrations depend on the selected application component and deployment configuration.*

---

## Project Structure

The repository is organized into application source code, deployment assets, model infrastructure, and supporting configuration.

```text
Enterprise-data-assistant/
│
├── cri/
│   ├── config/
│   ├── core/
│   ├── data/
│   ├── frontend/
│   ├── scripts/
│   └── ...
│
├── docmind/
│   └── ...
│
├── llama-server/
│   └── ...
│
├── models/
│   └── ...
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

**Directory descriptions**

| Directory / File | Purpose |
|---|---|
| `cri/` | Source code and supporting resources for the enterprise data assistant and NL2SQL workflows. |
| `docmind/` | Source code for document-based question answering and knowledge retrieval. |
| `llama-server/` | Supporting local LLM inference infrastructure. |
| `models/` | Location for model assets when required by the deployment. |
| `Dockerfile` | Container build instructions, where applicable. |
| `docker-compose.yml` | Docker service configuration, if present in the deployment. |
| `.dockerignore` | Excludes selected files from Docker build contexts. |
| `.gitignore` | Excludes selected files from Git tracking. |

*Some supporting files and directories may be omitted from version control or differ between deployment environments.*

---

## How It Works

### CRI: Enterprise Data Querying

1. A user submits a question in natural language.
2. The application identifies the relevant enterprise data context.
3. Schema metadata and table relationships help guide SQL generation.
4. The configured LLM generates a candidate SQL query.
5. SQL validation checks the generated query against application rules.
6. The query is processed through the configured database integration.
7. The application returns the resulting information to the user.

### DocMind: Document Question Answering

1. An administrator prepares and uploads supported documents through the configured document-management workflow.
2. Documents are processed for retrieval.
3. Text representations and embeddings are used to support semantic search.
4. A user selects the relevant knowledge module and submits a question.
5. The retrieval pipeline identifies relevant document context.
6. The retrieved context is supplied to the configured LLM.
7. The application generates a context-aware response.

---

## Prerequisites

Before deploying the application, ensure the environment has the dependencies required by your selected configuration.

- Windows Server or another supported operating system.
- Docker and the appropriate container runtime.
- Python for local development and troubleshooting.
- Access to the configured relational database.
- Access to Qdrant when vector retrieval is enabled.
- Access to Neo4j when graph-based relationship processing is enabled.
- A configured LLM inference service.
- Required Python dependencies and application configuration.
- Network connectivity between the application and its dependent services.

Some components may run in containers, while others may use separately hosted services. Configure the required connectivity for your environment.

---

## Configuration

The application uses environment variables and configuration files to connect to databases, AI services, and supporting infrastructure.

Typical configuration categories include:

| Configuration | Purpose |
|---|---|
| Database connection | Database host, port, name, username, and password. |
| LLM configuration | Model name, inference endpoint, and generation settings. |
| Qdrant configuration | Vector database endpoint and authentication, if enabled. |
| Neo4j configuration | Graph database connection settings, if enabled. |
| Cache configuration | Cache server connection and related settings. |
| API configuration | Application ports and service endpoints. |

### Configuration Guidelines

1. Configure environment-specific settings outside the source code.
2. Use a local `.env` file or the deployment platform's secret-management mechanism where appropriate.
3. Keep `.env` files and credentials out of Git.
4. Do not commit production database passwords, API keys, access tokens, or private certificates.
5. Use separate configurations for development, testing, and production.
6. Confirm that application containers can reach their configured dependencies.

**Important:** Refer to the configuration files and deployment definitions in your environment for the exact variable names and required values. Do not copy production credentials into this repository.

---

## Docker Deployment

The application is designed to support containerized deployment.

### 1. Clone the Repository

```bash
git clone https://github.com/shree31-sudo/Enterprise-data-assistant.git
cd Enterprise-data-assistant
```

Repository access requires appropriate GitHub permissions.

### 2. Review the Deployment Configuration

Inspect the Dockerfiles and any available Docker Compose configuration.

Confirm that:

- Required application files are available.
- Environment variables are configured.
- Database and AI service endpoints are correct.
- Required ports are available.
- Persistent storage is configured where necessary.

### 3. Configure Dependencies

Ensure that required services, such as the relational database, vector database, graph database, caching infrastructure, and LLM inference service, are reachable from the application.

### 4. Build and Run

Use the build and startup commands defined by the actual deployment configuration.

For a service defined in a Docker Compose file, a typical build command is:

```bash
docker compose build
```

A typical startup command is:

```bash
docker compose up -d
```

Run these commands only after reviewing the Compose configuration and confirming the target environment.

**For an existing production deployment:** Do not rebuild or restart running services solely to update the GitHub repository. Source control and container deployment are separate operations.

---

## Security Considerations

Enterprise Data Assistant may connect to internal databases, business metadata, technical documentation, and other enterprise resources.

Follow these practices when developing and deploying the application:

- Keep credentials out of source code and version control.
- Restrict database access to the minimum permissions required.
- Protect application endpoints with appropriate authentication and authorization.
- Validate generated SQL before execution.
- Apply appropriate controls to database queries and user input.
- Restrict access to internal documentation and knowledge collections.
- Avoid exposing internal service endpoints publicly.
- Review configuration and metadata files before sharing the repository.
- Use HTTPS and appropriate network controls for deployed services.
- Review application logs to avoid recording sensitive information.
- Rotate credentials immediately if they are accidentally exposed.

**Repository privacy is not a substitute for secret management or authorization to publish company-owned code.**

---

## Development Workflow

A typical development workflow is:

1. Review the existing application architecture.
2. Implement changes within the appropriate component.
3. Test the affected functionality.
4. Validate database connectivity and AI service integration.
5. Review configuration changes for sensitive information.
6. Commit the changes to Git.
7. Push the changes to the authorized GitHub repository.
8. Deploy through the approved deployment process.

Example Git workflow:

```bash
git status
git add .
git commit -m "Describe your changes"
git push
```

Review the staged changes before committing to ensure that secrets, generated artifacts, internal datasets, and unintended files are not included.

---

## Use Cases

### Enterprise Analytics
Retrieve business information by asking questions in natural language.

### Intelligent Database Access
Generate SQL from natural language with schema-aware context and relationship information.

### Technical Documentation Assistant
Ask questions about technical manuals and retrieve contextually relevant information.

### Knowledge Management
Make document-based enterprise knowledge easier to search and access.

### AI-Assisted Information Retrieval
Combine semantic search and LLM-based generation to improve access to structured and unstructured information.

---

## Future Enhancements

Potential areas for further development include:

- Improved SQL generation accuracy and validation.
- Expanded support for database schemas and relationship types.
- More efficient semantic retrieval and document processing.
- Enhanced source attribution for document-based responses.
- Improved observability and performance monitoring.
- Fine-grained access control for enterprise knowledge.
- Automated testing and deployment pipelines.
- Improved handling of ambiguous natural language questions.

These are potential enhancements rather than a statement that every item is currently implemented.

---

## Repository

**GitHub:**  
https://github.com/shree31-sudo/Enterprise-data-assistant

This repository is intended for authorized development and maintenance of the Enterprise Data Assistant project.

---

## License

No open-source license has been specified.

If this project contains company-owned source code, obtain the appropriate authorization before distributing it or adding an open-source license.

---

## Acknowledgements

Built using modern AI, backend, database, information retrieval, and containerization technologies to support intelligent enterprise data access and knowledge discovery.