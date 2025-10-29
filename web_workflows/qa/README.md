# Web-based QA Workflow Visualization

This web interface lets you visualize each step of the PikeRAG QA workflow with your earthquakes knowledge base.

## Features

🔍 **Retrieval Visualization**: See which chunks are retrieved from your knowledge base  
✍️ **Prompt Inspection**: View the complete formatted prompt sent to the LLM  
🤖 **LLM Response**: See the raw response from the language model  
📝 **Parsing Results**: Check how the output is parsed and structured  
✅ **Final Answers**: Compare answers with expected results

## Quick Start

1. **Activate your virtual environment**:
```bash
source /home/lkm/Pictures/PIKE-RAG/test_env/bin/activate
```

2. **Install Flask** (if not already installed):
```bash
pip install flask
```

3. **Run the web server**:
```bash
cd /home/lkm/Videos/pike_rag
python web_workflows/qa/app.py
```

4. **Open your browser**:
```
http://localhost:5000
```

## Usage

1. **Load Configuration**: 
   - Enter the path to your YAML config (default: `examples/earthquakes/configs/qa.yml`)
   - Click "Load Config"

2. **Select a Question**:
   - Browse the list of questions
   - Click on any question to process it

3. **View the Workflow**:
   - See all 5 steps of the QA process
   - Expand sections to see detailed information
   - Check the retrieved chunks, full prompts, LLM responses, and final answers

## Architecture

```
web_workflows/qa/
├── app.py              # Flask backend - handles API requests
├── templates/
│   └── index.html      # Web UI with interactive visualization
└── README.md           # This file
```

## API Endpoints

- `POST /load_config`: Load a YAML config file
- `GET /get_questions`: Get list of questions from testing suite  
- `POST /answer_question`: Process a question and return debug steps
- `GET /health`: Health check

## For Beginners

This interface shows you:

1. **What chunks are retrieved** - Which parts of your document are used for answering
2. **How the prompt is built** - The exact instructions and context sent to the LLM
3. **What the LLM responds** - The raw model output before parsing
4. **How it's interpreted** - How the response is parsed into structured answers
5. **Whether it's correct** - Final answer compared to ground truth

Perfect for understanding how RAG works under the hood!

## Extending to Other Workflows

To add support for other workflows (IRCoT, Self-Ask, etc.):

1. Create subdirectories: `web_workflows/ircot/`, `web_workflows/self_ask/`, etc.
2. Follow the same structure as the `qa/` folder
3. Create workflow-specific visualization components

