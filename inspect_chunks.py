#!/usr/bin/env python3
import pickle
import glob
from pathlib import Path

# Inspect all pickle files
for pkl_file in glob.glob('data/earthquakes/chunks/*.pkl'):
    print(f'\n{"="*80}')
    print(f'File: {pkl_file}')
    print(f'{"="*80}')
    
    with open(pkl_file, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f'Total chunks: {len(chunks)}')
    
    # Show first few chunks
    for i, chunk in enumerate(chunks[:5]):
        print(f'\n--- Chunk {i} ---')
        print(f'Length: {len(chunk.page_content)} characters')
        print(f'Content: {repr(chunk.page_content[:200])}')
        if hasattr(chunk, 'metadata'):
            print(f'Metadata: {chunk.metadata}')
