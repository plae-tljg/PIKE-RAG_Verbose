# Manual Chunking Guide

Since LLM-based chunking takes time and can be expensive, this guide shows you how to manually create chunks with summaries.

## Quick Start

1. **Prepare your chunks** - Edit `create_manual_chunks.py` and add your chunk definitions
2. **Run the script** - `python create_manual_chunks.py`
3. **Done!** - Pickle files will be saved to `data/earthquakes/chunks/`

## Understanding the Format

Each chunk needs:
- **content**: The actual text of the chunk
- **summary**: Your manual summary describing what the chunk is about
- **source**: The original file path (for metadata)

### Example

```python
MANUAL_CHUNKS = {
    "ch5.txt": {
        "source_filename": "ch5.txt",
        "chunks": [
            {
                "content": "Volcanoes and Earthquakes\n\nEarthquakes are associated...",
                "summary": "This part discusses volcanic eruptions and earthquakes...",
                "source": "data/earthquakes/contents/ch5.txt"
            }
        ]
    }
}
```

## Workflow Options

### Option 1: Manual Entry (Fully Manual)

1. Read your source files from `data/earthquakes/contents/`
2. Decide how to split them into chunks (aim for 300-800 chars per chunk)
3. Write summaries for each chunk
4. Edit `create_manual_chunks.py` with your chunks
5. Run the script

### Option 2: Auto-Generate Template

Use the helper script to auto-split at paragraph boundaries:

```bash
python read_and_prepare_chunks.py
```

This will:
- Read all content files
- Try to split at paragraph boundaries (double newlines)
- Generate a template with TODO summaries
- Copy the template into `create_manual_chunks.py`
- You edit the summaries
- Run `create_manual_chunks.py`

### Option 3: Fully Manual with Helper

```bash
python helper_read_content_for_chunks.py
```

This shows you the content so you can decide manually where to split.

## Chunking Guidelines

### When to Split

Good places to split:
- Between distinct topics/subjects
- At paragraph breaks
- When content naturally has 2-3 clear subsections
- After 600-800 characters of related content

Avoid splitting:
- In the middle of a sentence
- In the middle of a thought/idea
- If the chunk would be less than 200 chars (merge with adjacent chunk)

### Summary Guidelines

Write summaries that:
- Start with "The main content of this part is..."
- Describe the key information in the chunk
- Are 1-3 sentences
- Would help someone know if this chunk is relevant

Examples:
```
"The main content of this part is the discussion on the recurrence of earthquakes along fault lines, the incomplete relief of stress even after an earthquake, and the potential for increased stress in other parts of the fault due to stress redistribution."

"The main content of this part describes how volcanic ash clouds damage aircraft engines and the formation of an international early warning system."
```

### Recommended Chunk Sizes

- **Minimum**: 200 characters (to ensure meaningful chunks)
- **Optimal**: 400-800 characters (good balance)
- **Maximum**: 1500 characters (don't make chunks too long)

## Complete Example

For ch5.txt (which is very short), here's a complete working example:

```python
"ch5.txt": {
    "source_filename": "ch5.txt",
    "chunks": [
        {
            "content": "Volcanoes and Earthquakes\n\nEarthquakes are associated with volcanic eruptions. Abrupt increases in earthquake activity heralded eruptions at Mount St. Helens, Washington; Mount Spurr and Redoubt Volcano, Alaska; and Kilauea and Mauna Loa, Hawaii. The location and movement of swarms of tremors indicate the movement of magma through the volcano. Continuous records of seismic and tiltmeter (a device that measures ground tilting) data are maintained at U.S. Geological Survey volcano observatories in Hawaii, Alaska, California, and the Cascades, where study of these records enables specialists to make short-range predictions of volcanic eruptions. These warnings have been especially effective in Alaska, where the imminent eruption of a volcano requires the rerouting of international air traffic to enable airplanes to avoid volcanic clouds. Since 1982, at least seven jumbo jets, carrying more than 1,500 passengers, have lost power in the air after flying into clouds of volcanic ash. Though all flights were able to restart their engines eventually and no lives were lost, the aircraft suffered damages of tens of millions of dollars. As a result of these close calls, an international team of volcanologists, meteorologists, dispatchers, pilots, and controllers have begun to work together to alert each other to imminent volcanic eruptions and to detect and track volcanic ash clouds.",
            "summary": "The main content of this part discusses the relationship between earthquakes and volcanic eruptions, providing examples from Mount St. Helens, Mount Spurr, Redoubt Volcano, Kilauea, and Mauna Loa. It explains how seismic and tiltmeter data help predict volcanic eruptions, the impact on air traffic from volcanic ash clouds, including incidents where jumbo jets lost power, and the international collaboration to improve early warning systems.",
            "source": "data/earthquakes/contents/ch5.txt"
        }
    ]
},
```

## Running the Script

```bash
python create_manual_chunks.py
```

Expected output:
```
================================================================================
Manual Chunk Creator for PIKE-RAG
================================================================================

Output directory: data/earthquakes/chunks
Found 1 files to process

Processing: ch5.txt
  Created 1 chunks:
    Chunk 1: 906 chars
            Summary: The main content of this part discusses...
✓ Saved 1 chunks to data/earthquakes/chunks/ch5.pkl

================================================================================
✓ All files processed successfully!
================================================================================
```

## Verification

Check the generated files:

```bash
ls -lh data/earthquakes/chunks/
# Should see .pkl files

# Optional: verify the pickled data
python -c "
import pickle
with open('data/earthquakes/chunks/ch5.pkl', 'rb') as f:
    chunks = pickle.load(f)
    print(f'Loaded {len(chunks)} chunks')
    for i, chunk in enumerate(chunks, 1):
        print(f'Chunk {i}: {len(chunk.page_content)} chars')
        print(f'Summary: {chunk.metadata[\"summary\"][:80]}...')
"
```

## Tips

1. **For very short files** (like ch5.txt): Keep as a single chunk
2. **For longer files**: Split at topic boundaries (don't force it)
3. **Trust your judgment**: If content naturally fits together, keep it together
4. **Quality over quantity**: Better 5 good chunks than 10 forced chunks

## Files Created

The script creates pickle files that are **identical in format** to what the LLM chunking workflow produces:
- Same Document structure
- Same metadata format
- Same pickle file format
- Ready to use in downstream QA workflows

## Troubleshooting

**"ImportError: langchain_core not found"**
```bash
pip install langchain_core
```

**"ModuleNotFoundError: langchain"**
```bash
pip install langchain
```

**"Permission denied"**
```bash
chmod +x create_manual_chunks.py
```

## Need Help?

- Read the source content: `python helper_read_content_for_chunks.py`
- Auto-generate templates: `python read_and_prepare_chunks.py`
- Edit chunks manually: `create_manual_chunks.py`

The key is: **write summaries that describe what information is in each chunk**, as these summaries will be used for retrieval.

