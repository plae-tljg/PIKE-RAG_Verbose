# Chunking Script Debug Information

This document explains the logging that has been added to help you understand the chunking procedure and output.

## What Was Modified

### Files Modified:
1. `pikerag/document_transformers/splitter/llm_powered_recursive_splitter.py` - Added detailed logging at each LLM interaction step
2. `pikerag/workflows/chunking.py` - Added summary logging showing final output

## How the Chunking Process Works

The LLM-powered chunking process follows these steps:

### STEP 1: Getting First Chunk Summary
- **Purpose**: Generate an initial summary of the first chunk of the document
- **What's logged**:
  - Text being summarized (first ~500 chars)
  - Message sent to LLM
  - Raw LLM response
  - Parsed summary extracted from response

### STEP 2: Resplit Chunk and Generate Summaries
- **Purpose**: Split text into two parts with contextual summaries
- **What's logged**:
  - Text to be split
  - Current chunk summary context
  - Message sent to LLM
  - Raw LLM response
  - Parsed result showing:
    - First chunk length
    - First chunk summary
    - Second chunk summary (for remaining text)
    - Dropped length (how much text was consumed)

### STEP 3: Getting Last Chunk Summary
- **Purpose**: Finalize the last chunk of the document
- **What's logged**:
  - Chunk content
  - Current chunk summary context
  - Message sent to LLM
  - Raw LLM response
  - Parsed final summary

### Final Results
Each chunk is saved with:
- **Content**: The actual text of the chunk
- **Summary**: A structured summary of the chunk's content
- **Metadata**: Including filename and other metadata

## Output Format

The chunks are saved as a list of Document objects, each containing:
```python
{
    "page_content": "the actual chunk text",
    "metadata": {
        "filename": "original_file_name",
        "summary": "the LLM-generated summary"
    }
}
```

These are pickled and saved to the output directory.

## Running the Script

When you run:
```bash
python examples/chunking.py examples/earthquakes/configs/chunking.yml
```

You will see:
1. Document processing information
2. Each LLM call with input/output
3. Final chunk summaries
4. File save confirmation

## Understanding the Output

- **[MESSAGE TO LLM]**: Shows the structured prompt with placeholders filled
- **[RESPONSE FROM LLM]**: Raw response from the language model
- **[PARSED RESULT/SUMMARY]**: Extracted information after parsing the LLM response
- **[FINAL CHUNK]**: Information about each completed chunk

The final summary shows all chunks with their content lengths, summaries, and metadata.
