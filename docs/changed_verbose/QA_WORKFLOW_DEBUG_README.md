# QA Workflow Debug Information

This document explains the logging that has been added to help you understand the QA workflow procedure and output.

## What Was Modified

### Files Modified:
1. `pikerag/workflows/qa.py` - Added detailed logging at each QA step
2. `examples/earthquakes/configs/qa.yml` - Updated embedding model from BioLORD to BAAI/bge-m3

## About the Embedding Model

**Original**: `FremyCompany/BioLORD-2023`
- BioLORD is a **biomedical domain-specific** embedding model
- Optimized for biomedical and health-related text
- Not ideal for general-purpose earthquake data

**Updated**: `BAAI/bge-m3`
- **BAAI/bge-m3** is a multilingual general-purpose embedding model
- Works well for general domain text including earthquake data
- Better choice for non-biomedical domains

**Other alternatives you can consider**:
- `sentence-transformers/all-mpnet-base-v2` - High quality general-purpose embeddings
- `intfloat/multilingual-e5-base` - Excellent multilingual support
- `BAAI/bge-large-en-v1.5` - Large English-only model for better quality

## How the QA Process Works

The QA workflow follows these steps for each question:

### STEP 1: Retrieve Relevant Chunks

This step involves multiple components:

#### 1.1: Query Generation
- **Purpose**: Parse the question to generate retrieval queries
- **Location**: `pikerag/knowledge_retrievers/query_parsers/qa_parser.py`
- **What's logged**:
  - Number of queries generated
  - Each query text

#### 1.2: Vector Store Retrieval
- **Purpose**: Search the vector store for relevant chunks
- **Location**: `pikerag/knowledge_retrievers/mixins/chroma_mixin.py`
- **What's logged**:
  - Number of raw results from similarity search
  - Number of results after score threshold filtering
  - Query used for search
  - retrieve_k and score_threshold values

#### 1.3: Retrieve Results with Scores
- **Location**: `pikerag/knowledge_retrievers/chroma_qa_retriever.py`
- **What's logged**:
  - Each chunk's similarity score
  - Chunk length and metadata
  - Chunk content previews (first 150 chars)
  - Total number of chunks retrieved per query

### STEP 2: Format Messages for LLM

#### 2.1: Prompt Encoding
- **Purpose**: Encode the question, options, and references into the LLM prompt format
- **Location**: `pikerag/prompts/qa/multiple_choice.py` or `pikerag/prompts/qa/generation.py`
- **What's logged**:
  - Number of references being encoded
  - Preview of first 3 references
  - Total context length
  - Number of references used (if truncated)

#### 2.2: Message Construction
- **What's logged**:
  - Complete message structure sent to the LLM
  - Includes system/user roles and formatted content
  - All placeholders filled in

### STEP 3: Call LLM
- **Purpose**: Send the formatted messages to the language model
- **What's logged**:
  - Raw response from the LLM

### STEP 4: Parse LLM Output
- **Purpose**: Extract structured information from the LLM's response
- **What's logged**:
  - Parsed output dictionary with extracted data

### STEP 5: Final Result
- **Purpose**: Display the final answer and compare with correct answer
- **What's logged**:
  - Predicted answer
  - Correct label/expected answers
  - Comparison results

## Question Processing Flow

For each question, you'll see detailed logging from multiple components:

```
================================================================================
QUESTION N
================================================================================

[QUESTION]
The actual question text

[OPTIONS] (for multiple choice)
A. Option 1
B. Option 2
...

================================================================================

[STEP 1: RETRIEVE RELEVANT CHUNKS]
================================================================================

[RETRIEVER] Starting retrieval for Q001
[RETRIEVER] Question: What is...
[RETRIEVER] Generated 1 query/queries:
  Query 1: What is earthquake magnitude?

[RETRIEVER] Will retrieve top-16 chunks per query

[RETRIEVER] Processing Query 1/1

[RETRIEVER] Query: What is earthquake magnitude?
[RETRIEVER] Searching for top-16 chunks with score threshold: 0.2

[SIMILARITY_SEARCH] Retrieved 16 results from vector store
[SIMILARITY_SEARCH] After filtering (threshold=0.2): 12 documents

[RETRIEVER] Found 12 matching chunks
  Chunk 1: Score=0.8542, Length=320, Metadata={'filename': 'ch1.txt', 'chunk_idx': 5}
  Chunk 2: Score=0.7891, Length=298, Metadata={'filename': 'ch2.txt', 'chunk_idx': 2}
  ...

[RETRIEVER] Chunk contents (first 150 chars of each):
  Chunk 1: Earthquake magnitude is a measure of the energy released...
  Chunk 2: The Richter scale measures...
  ...

[RETRIEVER] Total 12 chunks retrieved for Q001
✓ Retrieved 12 reference chunks
================================================================================

[STEP 2: FORMAT MESSAGES FOR LLM]
--------------------------------------------------------------------------------

[PROMPT_ENCODER] Encoding 12 references into prompt
  Reference 1: Earthquake magnitude is a measure of...
  Reference 2: The Richter scale measures...
  Reference 3: Magnitude is determined using...
  ... (9 more references)

Messages to LLM: [
  {'role': 'system', 'content': '...'},
  {'role': 'user', 'content': '...'}
]

[STEP 3: CALL LLM]
--------------------------------------------------------------------------------
Raw LLM Response: <result>\n<thinking>...</thinking>...

[STEP 4: PARSE LLM OUTPUT]
--------------------------------------------------------------------------------
Parsed Output: {'answer': 'A', 'thinking': '...'}

[STEP 5: FINAL RESULT]
================================================================================
Answer: A
Correct label: [A]
================================================================================
```

## Running the QA Workflow

When you run:
```bash
python examples/qa.py examples/earthquakes/configs/qa.yml
```

You will see:
1. Each question being processed
2. Retrieved chunks for each question
3. LLM messages and responses
4. Parsed results
5. Final answers vs correct answers
6. Overall accuracy metrics at the end

## Output Format

The QA workflow:
1. **Processes each question** with detailed step-by-step logging
2. **Retrieves relevant chunks** using semantic similarity
3. **Sends formatted prompts** to the LLM with context
4. **Extracts answers** from LLM responses
5. **Evaluates performance** using metrics like Exact Match, F1, etc.

Results are saved to:
- `logs/earthquakes/qa/qa_with_chunk_reference/test_results.jsonl`
- CSV files with metrics
- Log files with detailed information

## Understanding the Workflow

The QA system uses a **retrieval-augmented generation (RAG)** approach:
1. Questions are used to query the vector store (containing chunked documents)
2. Top-K most relevant chunks are retrieved
3. LLM is given the question + retrieved context
4. LLM generates an answer based on the context
5. Answer is compared against ground truth for evaluation

## Data Flow

```
Question → Query Parser → Vector Store Retrieval → 
Retrieved Chunks → Prompt Template → LLM → 
Parsed Answer → Evaluation → Results
```

## Notes

- The system automatically loads the test suite from the configured source
- Multiple rounds of testing can be configured via `test_rounds` in the YAML
- Results are cached to avoid redundant LLM calls
- Parallel processing is supported for faster evaluation

