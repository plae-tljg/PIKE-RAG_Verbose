#!/usr/bin/env python3
"""
Manual Chunk Creator for PIKE-RAG

This script allows you to manually create chunks with summaries and save them
in the same pickled format that the LLM-based chunking workflow produces.

Usage:
    1. Edit the MANUAL_CHUNKS dictionary below to define your chunks
    2. Run: python create_manual_chunks.py
    3. Pickle files will be saved to data/earthquakes/chunks/

Format:
Each entry should have:
    - filename: output filename (will become filename.pkl)
    - chunks: list of chunk dictionaries with:
        - content: the actual text content of the chunk
        - summary: your manual summary of the chunk
        - source: original source file path (optional)
"""

import pickle
from pathlib import Path
from typing import Dict, List

# Import Document from langchain
try:
    from langchain_core.documents import Document
except ImportError:
    print("langchain_core not found. Install with: pip install langchain_core")
    exit(1)


def create_chunks_from_manual_data(chunk_data: Dict) -> List[Document]:
    """
    Create Document objects from manual chunk definitions.
    
    Args:
        chunk_data: Dictionary with 'chunks' list containing chunk dicts
        
    Returns:
        List of Document objects ready for pickling
    """
    documents = []
    
    for chunk_info in chunk_data['chunks']:
        # Create metadata
        metadata = {
            'filename': chunk_data.get('source_filename', ''),
            'summary': chunk_info['summary'],
        }
        
        # Add optional source path
        if 'source' in chunk_info:
            metadata['source'] = chunk_info['source']
        
        # Create Document object
        doc = Document(
            page_content=chunk_info['content'],
            metadata=metadata
        )
        documents.append(doc)
        
    return documents


def save_chunks(chunks: List[Document], output_path: str) -> None:
    """Save chunks to pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"✓ Saved {len(chunks)} chunks to {output_path}")


# ============================================================================
# MANUAL CHUNK DEFINITIONS
# ============================================================================
# Edit this section to define your chunks manually

MANUAL_CHUNKS = {
    "ch5.txt": {
        "source_filename": "ch5.txt",
        "chunks": [
            {
                "content": """Volcanoes and Earthquakes  

Earthquakes are associated with volcanic eruptions. Abrupt increases in earthquake activity heralded eruptions at Mount St. Helens, Washington; Mount Spurr and Redoubt Volcano, Alaska; and Kilauea and Mauna Loa, Hawaii. The location and movement of swarms of tremors indicate the movement of magma through the volcano. Continuous records of seismic and tiltmeter (a device that measures ground tilting) data are maintained at U.S. Geological Survey volcano observatories in Hawaii, Alaska, California, and the Cascades, where study of these records enables specialists to make short-range predictions of volcanic eruptions. These warnings have been especially effective in Alaska, where the imminent eruption of a volcano requires the rerouting of international air traffic to enable airplanes to avoid volcanic clouds. Since 1982, at least seven jumbo jets, carrying more than 1,500 passengers, have lost power in the air after flying into clouds of volcanic ash. Though all flights were able to restart their engines eventually and no lives were lost, the aircraft suffered damages of tens of millions of dollars. As a result of these close calls, an international team of volcanologists, meteorologists, dispatchers, pilots, and controllers have begun to work together to alert each other to imminent volcanic eruptions and to detect and track volcanic ash clouds.""",
                "summary": "The main content of this part discusses the relationship between earthquakes and volcanic eruptions, providing examples from Mount St. Helens, Mount Spurr, Redoubt Volcano, Kilauea, and Mauna Loa. It explains how seismic and tiltmeter data help predict volcanic eruptions, the impact on air traffic from volcanic ash clouds, including incidents where jumbo jets lost power, and the international collaboration to improve early warning systems.",
                "source": "data/earthquakes/contents/ch5.txt"
            }
        ]
    },
    
    # Add more files here following this format:
    # "filename.txt": {
    #     "source_filename": "filename.txt",
    #     "chunks": [
    #         {
    #             "content": "Your first chunk content here...",
    #             "summary": "Your summary for this chunk",
    #             "source": "data/earthquakes/contents/filename.txt"
    #         },
    #         {
    #             "content": "Your second chunk content here...",
    #             "summary": "Your summary for this chunk",
    #             "source": "data/earthquakes/contents/filename.txt"
    #         }
    #     ]
    # },
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Process all manual chunk definitions and save pickle files."""
    
    # Output directory
    output_dir = Path("data/earthquakes/chunks")
    
    print("=" * 80)
    print("Manual Chunk Creator for PIKE-RAG")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Found {len(MANUAL_CHUNKS)} files to process\n")
    
    for filename, data in MANUAL_CHUNKS.items():
        print(f"\nProcessing: {filename}")
        
        # Create Document objects from manual data
        chunks = create_chunks_from_manual_data(data)
        
        # Print summary
        print(f"  Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"    Chunk {i}: {len(chunk.page_content)} chars")
            print(f"            Summary: {chunk.metadata['summary'][:80]}...")
        
        # Save to pickle file
        output_file = output_dir / f"{Path(filename).stem}.pkl"
        save_chunks(chunks, output_file)
    
    print("\n" + "=" * 80)
    print("✓ All files processed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

