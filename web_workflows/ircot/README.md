# Web-based IRCoT Workflow Visualization

This web interface lets you visualize the IRCoT (Iterative Retrieval with Chain-of-Thought) workflow for your earthquakes knowledge base.

## Features

🔍 **Iterative Retrieval**: See how the system retrieves and refines information over multiple rounds  
💭 **Chain-of-Thought**: Follow the reasoning process as the system builds answers  
📊 **Visualization**: Watch the multi-step reasoning and retrieval process  
✅ **Open QA**: Answer questions with free-form responses (not multiple choice)

## Quick Start

```bash
cd /home/lkm/Videos/pike_rag
./web_workflows/ircot/start_server.sh
```

Then open http://localhost:5001

## IRCoT Overview

IRCoT (Iterative Retrieval with Chain-of-Thought) performs multiple rounds of:
1. **Retrieval**: Get relevant chunks from your knowledge base
2. **Reasoning**: Generate reasoning about the question
3. **Refinement**: Use the reasoning to retrieve better chunks
4. **Final Answer**: Synthesize information into an answer

## Differences from QA Workflow

- **QA Workflow**: Single retrieval → answer
- **IRCoT Workflow**: Multiple retrieval rounds with reasoning between each

## Test Data

IRCoT uses open-ended QA format (JSONL):
- `data/earthquakes/open_qa_test.jsonl` - Sample questions with expected answers

Format:
```json
{"question": "What causes earthquakes?", "answer": "..."}
```

## For Beginners

See how:
1. Initial retrieval gets relevant chunks
2. System generates reasoning about the question
3. Reasoning helps refine the next retrieval
4. Multiple rounds improve the answer quality

Perfect for understanding iterative RAG systems!

