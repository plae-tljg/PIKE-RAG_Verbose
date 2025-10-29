#!/usr/bin/env python3
"""
Web-based IRCoT Workflow with RAG visualization
Shows iterative retrieval and refinement process
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from flask import Flask, render_template, request, jsonify
import yaml

# Add parent directory to path and examples directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "examples"))  # For importing earthquakes.utils

from pikerag.utils.config_loader import load_dot_env
from pikerag.workflows.common import BaseQaData, GenerationQaData
from pikerag.workflows.qa_ircot import QaIRCoTWorkflow

app = Flask(__name__, template_folder='templates', static_folder='static')
app.template_folder = 'templates'

# Global workflow instance
workflow = None
current_config_path = None


class WebIRCoTWorkflow(QaIRCoTWorkflow):
    """Extended IRCoT Workflow that captures all steps for web visualization"""
    
    def __init__(self, yaml_config: dict) -> None:
        self.debug_steps = []
        super().__init__(yaml_config)
    
    def answer(self, qa: BaseQaData, question_idx: int) -> dict:
        """Override to capture all debug information for web display"""
        
        print(f"\n[WEB] Starting IRCoT question {question_idx + 1}")
        
        references: List[str] = []
        rationales: List[str] = []
        responses: List[str] = []
        final_answer: str = None
        
        for round in range(self._max_num_question):
            # Store round info
            round_info = {
                "step": f"round_{round + 1}",
                "round": round + 1,
                "rationales": list(rationales)
            }
            
            # Retrieve more chunks
            if len(rationales) == 0:
                query = qa.question
            else:
                query = rationales[-1]
                
            round_info["query"] = query
            chunks = self._retriever.retrieve_contents_by_query(query, retrieve_id=f"Q{question_idx}_R{round}")
            references.extend(chunks)
            round_info["retrieved_chunks"] = chunks
            round_info["total_references"] = len(references)

            # Call LLM to generate rationale or answer
            messages = self._ircot_protocol.process_input(
                qa.question, rationales=rationales, references=references, is_limit=False,
            )
            
            # Capture prompts
            import json
            round_info["messages"] = messages
            round_info["full_prompt"] = json.dumps(messages, indent=2)
            
            response = self._client.generate_content_with_messages(messages, **self.llm_config)
            responses.append(response)
            round_info["llm_response"] = response
            
            output_dict = self._ircot_protocol.parse_output(response)
            round_info["parsed_output"] = output_dict

            if output_dict["answer"] is not None:
                final_answer = output_dict["answer"]
                round_info["final_answer"] = final_answer
                self.debug_steps.append(round_info)
                break
            elif isinstance(output_dict["next_rationale"], str):
                rationales.append(output_dict["next_rationale"])
                round_info["new_rationale"] = output_dict["next_rationale"]
            else:
                self.debug_steps.append(round_info)
                break
            
            self.debug_steps.append(round_info)

        if final_answer is None:
            # Final call
            final_round_info = {
                "step": "final_answer",
                "round": len(self.debug_steps) + 1,
                "total_references": len(references),
                "rationales": list(rationales)
            }
            
            messages = self._ircot_protocol.process_input(
                qa.question, rationales=rationales, references=references, is_limit=True,
            )
            
            import json
            final_round_info["messages"] = messages
            final_round_info["full_prompt"] = json.dumps(messages, indent=2)
            
            response = self._client.generate_content_with_messages(messages, **self.llm_config)
            responses.append(response)
            final_round_info["llm_response"] = response
            
            output_dict = self._ircot_protocol.parse_output(response)
            final_answer = output_dict["answer"]
            final_round_info["parsed_output"] = output_dict
            final_round_info["final_answer"] = final_answer
            
            self.debug_steps.append(final_round_info)

        return {
            "answer": final_answer,
            "rationale": rationales,
            "references": references,
            "responses": responses,
        }


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    experiment_name = yaml_config["experiment_name"]
    log_dir = os.path.join(yaml_config["log_root_dir"], experiment_name)
    
    # Make log_dir absolute (relative to project root, not current working directory)
    if not os.path.isabs(log_dir):
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
    """Load an IRCoT workflow config"""
    global workflow, current_config_path
    
    data = request.json
    config_path = data.get('config_path')
    
    # If path is relative, make it relative to project root
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
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
        
        print(f"[WEB_APP] Creating WebIRCoTWorkflow instance...")
        workflow = WebIRCoTWorkflow(yaml_config)
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
        
        # Answer the question using IRCoT workflow
        output = workflow.answer(qa, question_id)
        
        return jsonify({
            "success": True,
            "debug_steps": workflow.debug_steps,
            "final_output": output
        })
    except Exception as e:
        import traceback
        return jsonify({"error": f"{str(e)}\n\n{traceback.format_exc()}"}), 500


@app.route('/answer_custom_question', methods=['POST'])
def answer_custom_question():
    """Answer a custom typed question using IRCoT"""
    global workflow
    
    if workflow is None:
        return jsonify({"error": "No config loaded"}), 400
    
    data = request.json
    question_text = data.get('question', '')
    
    if not question_text:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        # Create a QA object for the custom question
        custom_qa = GenerationQaData(
            question=question_text,
            answer_labels=[]  # No ground truth for custom questions
        )
        
        # Clear debug steps
        workflow.debug_steps = []
        
        # Answer using a dummy question index
        output = workflow.answer(custom_qa, 999)
        
        return jsonify({
            "success": True,
            "debug_steps": workflow.debug_steps,
            "final_output": output
        })
    except Exception as e:
        import traceback
        return jsonify({"error": f"{str(e)}\n\n{traceback.format_exc()}"}), 500


@app.route('/rebuild_vector_store', methods=['POST'])
def rebuild_vector_store():
    """Rebuild the vector store"""
    global workflow
    
    if workflow is None:
        return jsonify({"error": "No config loaded"}), 400
    
    try:
        # Reload the vector store
        workflow._retriever._load_vector_store()
        return jsonify({
            "success": True,
            "message": "Vector store reloaded"
        })
    except Exception as e:
        import traceback
        return jsonify({"error": f"{str(e)}\n\n{traceback.format_exc()}"}), 500


if __name__ == '__main__':
    print("Starting IRCoT Web Workflow Server...")
    print("Visit http://localhost:5001 to use the interface")
    app.run(debug=True, host='0.0.0.0', port=5001)

