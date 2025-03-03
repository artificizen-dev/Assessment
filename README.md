# RAG-Based AI Agent Assessment Task

## Overview
Build a knowledge-based AI support agent that uses RAG (Retrieval Augmented Generation) to answer customer queries. The system should demonstrate your ability to implement efficient document retrieval.

## Requirements

### Core RAG Components

1. Document Processing Pipeline
   - Implement document ingestion for PDF and markdown files
   - Create efficient text chunking strategies
   - Generate and store embeddings using a vector store
   - Track document sources and maintain metadata

2. Retrieval System
   - Implement semantic search using embeddings
   - Create relevance scoring for retrieved chunks
   - Manage context window size effectively
   - Handle cases with multiple relevant documents

3. Response Generation
   - Generate coherent responses using retrieved context
   - Include source citations in responses
   - Handle cases where no relevant information is found
   - Ensure response accuracy against source material

### Technical Requirements

#### Vector Store Implementation
- Use a vector database (ChromaDB, Pinecone, Qdrant, or Weaviate)
- Implement efficient embedding generation
- Create proper indexing structure
- Handle document updates

#### Backend Development
- Use FastAPI or Django REST Framework
- Create endpoints for:
  - Document ingestion
  - Query processing
  - Knowledge base management

### Knowledge Base
The system should handle:
- PDF documents
- Word Documents

## Deliverables

1. Source Code
   - Documented code with clear README
   - Setup instructions
   - Configuration examples
   - Data preprocessing scripts

2. Technical Documentation
   - RAG implementation details
   - Email integration approach
   - System architecture diagram
   - API documentation

## Time Allocation
- 1 Week for completion
- Submit on the provided GitHub repository

## Sample Test Cases
1. Process and query documents
2. Handle edge cases (no irrelevant info, multiple sources handling)
3. Show error handling

## Submission Requirements
1. GitHub repository with:
   - Complete source code on github
   - Maintain the version history
   - Instruction to run the code in the submission section
   - Loom video link in the submission section

2. Demo showing:
   - Document ingestion process
   - Query-response examples
   - Error handling scenarios

## Notes for Candidates
- Take any test dataset from the internet
- Focus on RAG quality
- Document your chunking strategy
- Explain context management approach
- Include ideas for future improvements
- Provide example queries and responses

# Submission
