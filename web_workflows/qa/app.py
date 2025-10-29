#!/usr/bin/env python3
"""
Web-based QA Workflow with RAG visualization
Shows each step: retrieval, prompt formatting, LLM response, and parsing
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import yaml

# Add parent directory to path and examples directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "examples"))  # For importing earthquakes.utils

from pikerag.utils.config_loader import load_dot_env
from pikerag.workflows.common import BaseQaData, MultipleChoiceQaData
from pikerag.workflows.qa import QaWorkflow

app = Flask(__name__, template_folder='templates', static_folder='static')

# Allow both index.html and chat.html
app.template_folder = 'templates'

# Global workflow instance
workflow = None
current_config_path = None


class WebQaWorkflow(QaWorkflow):
    """Extended QA Workflow that captures all steps for web visualization"""
    
    def __init__(self, yaml_config: dict) -> None:
        self.debug_steps = []
        self._original_protocol = None  # Store original protocol
        super().__init__(yaml_config)
        self._original_protocol = self._qa_protocol  # Save it
    
    def answer(self, qa, question_idx: int) -> dict:
        """Override to capture all debug information for web display"""
        
        # STEP 1: Retrieval
        print(f"\n[WEB] Starting question {question_idx + 1}")
        step1_info = {"step": "retrieval", "question": qa.question}
        
        if isinstance(qa, MultipleChoiceQaData):
            step1_info["options"] = qa.options
        
        # Retrieve chunks
        reference_chunks = self._retriever.retrieve_contents(qa, retrieve_id=f"Q{question_idx:03}")
        step1_info["retrieved_count"] = len(reference_chunks)
        step1_info["chunks"] = [chunk[:300] + "..." if len(chunk) > 300 else chunk for chunk in reference_chunks]  # Show ALL chunks
        self.debug_steps.append(step1_info)
        
        # STEP 2: Format prompt
        messages = self._qa_protocol.process_input(content=qa.question, references=reference_chunks, **qa.as_dict())
        step2_info = {
            "step": "prompt_formatting",
            "num_messages": len(messages),
            "system_message": messages[0]['content'] if messages else "",
            "user_message": messages[1]['content'] if len(messages) > 1 else "",
            "full_prompt": json.dumps(messages, indent=2)
        }
        self.debug_steps.append(step2_info)
        
        # STEP 3: Call LLM
        response = self._client.generate_content_with_messages(messages, **self.llm_config)
        step3_info = {
            "step": "llm_response",
            "raw_response": response
        }
        self.debug_steps.append(step3_info)
        
        # STEP 4: Parse output
        output_dict = self._qa_protocol.parse_output(response, **qa.as_dict())
        step4_info = {
            "step": "parsing",
            "extracted_fields": list(output_dict.keys()),
            "parsed_output": output_dict
        }
        self.debug_steps.append(step4_info)
        
        if "response" not in output_dict:
            output_dict["response"] = response
        if "reference_chunks" not in output_dict:
            output_dict["reference_chunks"] = reference_chunks
        
        step5_info = {
            "step": "final_result",
            "answer": output_dict.get('answer', 'N/A'),
            "mask": output_dict.get('mask', ''),
            "correct_labels": qa.answer_mask_labels if hasattr(qa, 'answer_mask_labels') else []
        }
        self.debug_steps.append(step5_info)
        
        return output_dict


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    experiment_name = yaml_config["experiment_name"]
    log_dir = os.path.join(yaml_config["log_root_dir"], experiment_name)
    
    # Make log_dir absolute (relative to project root, not current working directory)
    if not os.path.isabs(log_dir):
        project_root = Path(__file__).parent.parent.parent
        log_dir = os.path.join(project_root, log_dir)
    
    yaml_config["log_dir"] = log_dir
    print(f"[WEB_APP] Log directory set to: {log_dir}")
    print(f"[WEB_APP] Log directory exists: {os.path.exists(log_dir)}")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set test_jsonl_path
    if yaml_config["test_jsonl_filename"] is None:
        yaml_config["test_jsonl_filename"] = f"{experiment_name}.jsonl"
    yaml_config["test_jsonl_path"] = os.path.join(log_dir, yaml_config["test_jsonl_filename"])
    
    # LLM cache config
    if yaml_config["llm_client"]["cache_config"]["location_prefix"] is None:
        yaml_config["llm_client"]["cache_config"]["location_prefix"] = experiment_name
    
    return yaml_config


@app.route('/')
def index():
    """Main page"""
    return render_template('chat.html')


@app.route('/load_config', methods=['POST'])
def load_config():
    """Load a QA workflow config"""
    global workflow, current_config_path
    
    data = request.json
    config_path = data.get('config_path')
    
    # If path is relative, make it relative to project root
    if not os.path.isabs(config_path):
        project_root = Path(__file__).parent.parent.parent
        config_path = os.path.join(project_root, config_path)
    
    # Log the path being checked
    print(f"[DEBUG] Checking config path: {config_path}")
    print(f"[DEBUG] Absolute path: {os.path.abspath(config_path)}")
    print(f"[DEBUG] Exists: {os.path.exists(config_path)}")
    
    if not os.path.exists(config_path):
        error_msg = f"Config file not found: {config_path} (absolute: {os.path.abspath(config_path)})"
        print(f"[ERROR] {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    try:
        # Ensure examples directory is in sys.path for module imports
        examples_dir = project_root / "examples"
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        
        print(f"\n[WEB_APP] Loading config from: {config_path}")
        yaml_config = load_yaml_config(config_path)
        load_dot_env(yaml_config.get("dotenv_path", None))
        
        # Clear previous debug steps
        if hasattr(workflow, 'debug_steps'):
            workflow.debug_steps = []
        
        print(f"[WEB_APP] Creating WebQaWorkflow instance...")
        workflow = WebQaWorkflow(yaml_config)
        print(f"[WEB_APP] Workflow initialized successfully")
        
        current_config_path = config_path
        
        return jsonify({
            "success": True,
            "config": yaml_config,
            "message": f"Loaded config: {config_path}"
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Failed to load config: {error_details}")
        return jsonify({"error": f"{str(e)}\n\nDetails:\n{error_details}"}), 500


@app.route('/get_questions', methods=['GET'])
def get_questions():
    """Get list of questions from the loaded testing suite"""
    global workflow
    
    if workflow is None:
        return jsonify({"error": "No config loaded"}), 400
    
    try:
        questions = []
        for idx, qa in enumerate(workflow._testing_suite):
            questions.append({
                "id": idx,
                "question": qa.question,
                "options": qa.options if hasattr(qa, 'options') else {},
                "type": type(qa).__name__
            })
        
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/answer_question', methods=['POST'])
def answer_question():
    """Answer a single question and return all debug steps"""
    global workflow
    
    if workflow is None:
        return jsonify({"error": "No config loaded"}), 400
    
    data = request.json
    question_id = data.get('question_id')
    
    try:
        qa = workflow._testing_suite[question_id]
        
        # Clear debug steps
        workflow.debug_steps = []
        
        # Answer the question
        output = workflow.answer(qa, question_id)
        
        # Get all debug steps
        return jsonify({
            "success": True,
            "debug_steps": workflow.debug_steps,
            "final_output": output
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/answer_custom_question', methods=['POST'])
def answer_custom_question():
    """Answer a custom typed question"""
    global workflow
    
    if workflow is None:
        return jsonify({"error": "No config loaded"}), 400
    
    data = request.json
    question_text = data.get('question', '')
    
    if not question_text:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        # For custom questions, use GenerationQaData instead of MultipleChoiceQaData
        # This allows free-form answers without needing options
        from pikerag.workflows.common import GenerationQaData
        
        # Create a QA object for generation (free-form answer)
        custom_qa = GenerationQaData(
            question=question_text,
            answer_labels=[]  # No ground truth labels
        )
        
        # Temporarily switch to generation protocol for this question
        from pikerag.prompts.qa import generation_qa_with_reference_protocol
        from pikerag.knowledge_retrievers.query_parsers import question_as_query
        
        original_protocol = workflow._qa_protocol
        original_parser = workflow._retriever._query_parser
        
        # Switch to generation protocol
        workflow._qa_protocol = generation_qa_with_reference_protocol
        
        # Use simple question_as_query for generation (no options needed)
        workflow._retriever._query_parser = question_as_query
        
        # Clear debug steps
        workflow.debug_steps = []
        
        try:
            # Answer using a dummy question index
            output = workflow.answer(custom_qa, 999)
        finally:
            # Restore original protocol and parser
            workflow._qa_protocol = original_protocol
            workflow._retriever._query_parser = original_parser
        
        return jsonify({
            "success": True,
            "debug_steps": workflow.debug_steps,
            "final_output": output
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Failed to answer custom question: {error_details}")
        return jsonify({"error": f"{str(e)}\n\nDetails:\n{error_details}"}), 500


@app.route('/rebuild_vector_store', methods=['POST'])
def rebuild_vector_store():
    """Rebuild the vector store by forcing recreation"""
    global workflow
    
    if workflow is None:
        return jsonify({"error": "No config loaded. Please load a config first."}), 400
    
    try:
        # Force deletion of existing vector store
        retriever_config = workflow._retriever_config
        vector_store_config = retriever_config["vector_store"]
        log_dir = workflow._log_dir
        collection_name = vector_store_config.get("collection_name", "earthquakes_book")
        persist_directory = vector_store_config.get("persist_directory", log_dir)
        
        # Delete old vector store
        import shutil
        if os.path.exists(persist_directory):
            chroma_path = os.path.join(persist_directory, "chroma.sqlite3")
            if os.path.exists(chroma_path):
                os.remove(chroma_path)
                print(f"[REBUILD] Deleted existing ChromaDB: {chroma_path}")
            
            # Delete collection directories
            for item in os.listdir(persist_directory):
                item_path = os.path.join(persist_directory, item)
                if os.path.isdir(item_path) and len(item) == 36:  # UUID pattern
                    shutil.rmtree(item_path)
                    print(f"[REBUILD] Deleted collection directory: {item_path}")
        
        # Re-initialize workflow which will recreate vector store
        from pikerag.knowledge_retrievers.chroma_qa_retriever import load_vector_store_from_configs
        
        import pickle
        from langchain_core.documents import Document
        from pathlib import Path
        
        # Load chunks
        chunk_dir = vector_store_config["id_document_loading"]["args"]["chunk_file_dir"]
        chunk_files = list(Path(chunk_dir).glob("*.pkl"))
        
        # Rebuild the vector store
        workflow._retriever.vector_store = load_vector_store_from_configs(
            vector_store_config=vector_store_config,
            embedding_config=vector_store_config.get("embedding_setting", {}),
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        
        return jsonify({
            "success": True,
            "message": f"Vector store rebuilt. Collection: {collection_name}"
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Failed to rebuild vector store: {error_details}")
        return jsonify({"error": f"{str(e)}\n\nDetails:\n{error_details}"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    print("Starting QA Web Workflow Server...")
    print("Visit http://localhost:5000 to use the interface")
    app.run(debug=True, host='0.0.0.0', port=5000)

