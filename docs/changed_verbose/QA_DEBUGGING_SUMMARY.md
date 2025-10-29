# QA Workflow Debugging - Complete Summary

## Files Modified for Verbose Logging

### 1. Main QA Workflow
**File**: `pikerag/workflows/qa.py`
- Added question display with options
- Added 5-step logging for each question
- Shows: Retrieve → Format → Call LLM → Parse → Result

### 2. Retriever Components

**File**: `pikerag/knowledge_retrievers/chroma_qa_retriever.py`
- Added logging for query generation
- Shows all retrieval queries being generated
- Displays similarity scores for each retrieved chunk
- Shows chunk metadata and length
- Displays chunk content previews

**File**: `pikerag/knowledge_retrievers/mixins/chroma_mixin.py`
- Added logging for vector store similarity search
- Shows raw results before filtering
- Shows filtered results after score threshold
- Displays retrieve_k and threshold values

### 3. Prompt Encoding

**File**: `pikerag/prompts/qa/multiple_choice.py`
- Added logging in MultipleChoiceQaWithReferenceParser
- Shows number of references being encoded
- Displays previews of references (first 100 chars of first 3)
- Shows count of remaining references

**File**: `pikerag/prompts/qa/generation.py`
- Added logging in GenerationQaParser
- Shows context length limit
- Displays number of references used
- Shows total context length
- Indicates if truncation occurred

### 4. Configuration Update

**File**: `examples/earthquakes/configs/qa.yml`
- Changed embedding model from BioLORD-2023 to BAAI/bge-m3
- Added comments explaining the choice
- Listed alternative embedding models

## What You'll See During Execution

### Retrieval Phase Details
- Query generation process
- Vector store search parameters
- Similarity scores for each chunk
- Metadata about retrieved chunks
- Chunk content previews

### Prompt Construction Details
- How many references are being encoded
- Previews of the references
- Total context length
- Whether truncation occurred

### LLM Interaction
- Complete message structure before sending to LLM
- Raw LLM response
- Parsed output from response
- Final answer comparison with ground truth

## Key Benefits

1. **Full Transparency**: See exactly what data flows at each step
2. **Debug Retrieval**: Understand if chunks are relevant and why
3. **Debug Prompting**: See how references are formatted for LLM
4. **Debug Parsing**: Verify LLM responses are correctly extracted
5. **Performance Insights**: See similarity scores to understand retrieval quality

## File Locations Reference

- Main Workflow: `pikerag/workflows/qa.py`
- Retriever: `pikerag/knowledge_retrievers/chroma_qa_retriever.py`
- Vector Search: `pikerag/knowledge_retrievers/mixins/chroma_mixin.py`
- Multiple Choice Prompt: `pikerag/prompts/qa/multiple_choice.py`
- Generation Prompt: `pikerag/prompts/qa/generation.py`
- Query Parser: `pikerag/knowledge_retrievers/query_parsers/qa_parser.py`
- Config: `examples/earthquakes/configs/qa.yml`

## Running with Debug Output

```bash
python examples/qa.py examples/earthquakes/configs/qa.yml
```

All the verbose logging will appear in your terminal, showing the complete QA pipeline in action!

