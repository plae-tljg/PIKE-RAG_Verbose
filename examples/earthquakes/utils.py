# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from typing import List, Literal, Tuple

from datasets import load_dataset, Dataset
from tqdm import tqdm

from langchain_core.documents import Document

from pikerag.utils.walker import list_files_recursively
from pikerag.workflows.common import MultipleChoiceQaData, GenerationQaData


def load_testing_suite(path: str="cais/mmlu", name: str="college_biology") -> List[MultipleChoiceQaData]:
    dataset: Dataset = load_dataset(path, name)["test"]
    testing_suite: List[dict] = []
    for qa in dataset:
        testing_suite.append(
            MultipleChoiceQaData(
                question=qa["question"],
                metadata={
                    "subject": qa["subject"],
                },
                options={
                    chr(ord('A') + i): choice
                    for i, choice in enumerate(qa["choices"])
                },
                answer_mask_labels=[chr(ord('A') + qa["answer"])],
            )
        )
    return testing_suite


def load_ids_and_chunks(chunk_file_dir: str) -> Tuple[Literal[None], List[Document]]:
    chunks: List[Document] = []
    chunk_idx: int = 0
    for doc_name, doc_path in tqdm(
        list_files_recursively(directory=chunk_file_dir, extensions=["pkl"]),
        desc="Loading Files",
    ):
        with open(doc_path, "rb") as fin:
            chunks_in_file: List[Document] = pickle.load(fin)

        for doc in chunks_in_file:
            doc.metadata.update(
                {
                    "filename": doc_name,
                    "chunk_idx": chunk_idx,
                }
            )
            chunk_idx += 1

        chunks.extend(chunks_in_file)

    return None, chunks


def load_open_qa_test_suite(filepath: str) -> List[GenerationQaData]:
    """Load open-ended QA questions from JSONL file"""
    import json
    import os
    from pathlib import Path
    from pikerag.workflows.common import GenerationQaData
    
    # If path is relative, make it relative to project root
    if not os.path.isabs(filepath):
        # Get project root (assume we're in examples/earthquakes/, go up 2 levels)
        project_root = Path(__file__).parent.parent.parent
        filepath = project_root / filepath
    
    test_suite = []
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:  # Skip empty lines
                continue
            try:
                data = json.loads(stripped)
                metadata = data.get("metadata", {})
                metadata["id"] = data.get("id", "")
                metadata["question_type"] = metadata.get("question_type", "open_qa")
                
                test_suite.append(
                    GenerationQaData(
                        question=data["question"],
                        answer_labels=[str(label) for label in data.get("answer_labels", [])],
                        metadata=metadata,
                    )
                )
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping invalid JSON line: {stripped[:100]}... Error: {e}")
                continue
    return test_suite


def load_earthquakes_test_suite() -> List[MultipleChoiceQaData]:
    """Load earthquake-specific test questions"""
    test_suite = [
        MultipleChoiceQaData(
            question="Where do most earthquakes occur on Earth?",
            metadata={"topic": "earthquake_locations"},
            options={
                "A": "At plate boundaries where tectonic plates meet",
                "B": "In the center of continents far from coastlines",
                "C": "Only in regions with active volcanoes",
                "D": "Exclusively in the Pacific Ocean"
            },
            answer_mask_labels=["A"]
        ),
        MultipleChoiceQaData(
            question="What type of plate boundary is characterized by deep-ocean trenches, earthquakes at various depths, and volcanic mountain ranges?",
            metadata={"topic": "plate_boundaries"},
            options={
                "A": "Spreading zones",
                "B": "Transform faults",
                "C": "Subduction zones",
                "D": "Convergent faults"
            },
            answer_mask_labels=["C"]
        ),
        MultipleChoiceQaData(
            question="What causes an earthquake to happen?",
            metadata={"topic": "earthquake_mechanism"},
            options={
                "A": "Sudden dislocation of segments of the Earth's crust",
                "B": "Changes in atmospheric pressure",
                "C": "Ocean currents and tides",
                "D": "Magnetic field variations"
            },
            answer_mask_labels=["A"]
        ),
        MultipleChoiceQaData(
            question="Which type of seismic wave travels faster and reaches the surface first after an earthquake?",
            metadata={"topic": "seismic_waves"},
            options={
                "A": "S waves (shear waves)",
                "B": "P waves (compressional waves)",
                "C": "Surface waves",
                "D": "Tsunami waves"
            },
            answer_mask_labels=["B"]
        ),
        MultipleChoiceQaData(
            question="What does the Richter Scale measure?",
            metadata={"topic": "magnitude_measurement"},
            options={
                "A": "The intensity of ground shaking at a specific location",
                "B": "The amplitude of seismic waves",
                "C": "The duration of the earthquake",
                "D": "The distance from the epicenter"
            },
            answer_mask_labels=["B"]
        ),
        MultipleChoiceQaData(
            question="What is liquefaction during an earthquake?",
            metadata={"topic": "liquefaction"},
            options={
                "A": "The process of rocks melting under intense heat",
                "B": "When loosely packed, water-logged sediments lose their strength",
                "C": "The formation of new fault lines",
                "D": "The acceleration of volcanic activity"
            },
            answer_mask_labels=["B"]
        ),
        MultipleChoiceQaData(
            question="During the 1964 Alaska earthquake, what caused most of the destruction at Kodiak, Cordova, and Seward?",
            metadata={"topic": "1964_alaska_earthquake"},
            options={
                "A": "Direct ground shaking",
                "B": "Volcanic eruptions",
                "C": "Tsunamis (sea waves)",
                "D": "Avalanches"
            },
            answer_mask_labels=["C"]
        ),
        MultipleChoiceQaData(
            question="What is the focal depth of an earthquake?",
            metadata={"topic": "earthquake_depth"},
            options={
                "A": "The depth where the earthquake's energy originates",
                "B": "The height of tsunami waves",
                "C": "The distance between tectonic plates",
                "D": "The thickness of the Earth's crust"
            },
            answer_mask_labels=["A"]
        ),
        MultipleChoiceQaData(
            question="Which fault type occurs in response to pulling or tension, where the overlying block moves down?",
            metadata={"topic": "fault_types"},
            options={
                "A": "Thrust faults",
                "B": "Strike-slip faults",
                "C": "Normal faults",
                "D": "Reverse faults"
            },
            answer_mask_labels=["C"]
        ),
        MultipleChoiceQaData(
            question="In which year did the New Madrid earthquakes occur, which were among the most widely felt earthquakes in North American history?",
            metadata={"topic": "historical_earthquakes"},
            options={
                "A": "1789-90",
                "B": "1811-12",
                "C": "1886-87",
                "D": "1906-07"
            },
            answer_mask_labels=["B"]
        ),
    ]
    return test_suite
