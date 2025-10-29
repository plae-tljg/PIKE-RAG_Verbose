# Changed

References: [https://github.com/microsoft/PIKE-RAG](https://github.com/microsoft/PIKE-RAG), [https://github.com/MrYangPengJu/PIKE-RAG-simplified](https://github.com/MrYangPengJu/PIKE-RAG-simplified)  

Go to [README_Original.md](README_Original.md) if you want to see the original project README.  

This is a project where I make different output of PIKE-RAG to be more verbose, so that readers can understand without being obstructed by those `importlib` which is impossiblt to be resolved by IDE.  

## Running in terminal

```bash
export PYTHONPATH=$PWD

python examples/chunking.py examples/earthquakes/configs/chunking.yml

python examples/qa.py examples/earthquakes/configs/qa.yml
```

I manually created a database of earthquakes from ebook in [https://www.gutenberg.org/ebooks/47340](https://www.gutenberg.org/ebooks/47340).  

However note that if you directly copy the ebook may have problem (?), since many sentence are separated not by space but by newlines. (BTW, easy way to treat it is open gedit then replace `\n\n` by `{{lkm}}`, then replace `\n` by ` `, then replace `{{lkm}}` by `\n\n`)  

BTW, my txt for different chapters isnt quite good for chunking, in that some chunks are just letters like `uds`, `]`, ... You may consider a manual editing so as to be better for chunking. I have that problem but then I use AI generated script to edit it to delete those chunks that are too small, so the chunk provided doesnt have that problem very visibly.  

The sample output of chunking is in `log_test_chunk.txt`.  

The sample output of qa workflow is in `log_test_output_qa`.  


## Manual Generation of Chunking

If you have a knowledge base which is too large for LLM processing, you may want manually creating the chunk files directly. You can checkout the formats easily from the script `create_manual_chunks.py`, just edit it to make chunks nad place to suitable directory.  

## try web form

For qa workflow, (assuming you have already done qa workflow in terminal and generated `chroma.sqlite3` and relevant bin files. If not, you can try the button to regenerate db in the webpage, but use at your own risk, I never try it successfully)  

PS. if you have done such thing, you can actually copy-and-paste the data to other folder for other workflow easily.  

```bash
cd web_workflows/qa && python app.py
```

For ircot workflow, can use:  

```bash
cd web_workflows/ircot && python app.py
```

In these webpage, rememer click load config first.  

## Key Changes

1. I added in `qwen_client` a memory_mode since if your VRam is just `6GB, loading the embedding model and Instruct model at the same time can lead to `CUDA out of memory`. Change to 

    ```bash
	llm_client:
	  module_path: pikerag.llm_client
	  # available class_name: AzureMetaLlamaClient, AzureOpenAIClient, HFMetaLlamaClient, QwenClient
	  class_name: QwenClient
	  args:
	    # memory_mode: "persistent"  # Keep model in GPU (faster, uses more memory)
	    memory_mode: "unload_after_use"  # Unload after each use (slower, saves memory)

	  llm_config:
	    model: /home/lkm/Downloads/Qwen2.5-7B-Instruct
	    temperature: 0

	  cache_config:
	    # location_prefix: will be joined with log_dir to generate the full path;
	    #   if set to null, the experiment_name would be used
	    location_prefix: null
	    auto_dump: True
    ```

2. I make a change to `pikerag/knowledge_retrievers/query_parsers/qa_parser.py` the function `question_plus_options_as_query` since it seem to see a string as list of string, which causes error.  

3. I make some changes to the ircot workflow, you can see in commit since my small model of `Qwen2.5-7B` seems to answer directly without CoT and continuous retrievel of chunks. I modified the prompt so that it defaults to higher round of reasoning, change it if you dont like it.  

PS. you can look at `docs/changed_verbose` to see many of my verbose experiemnt result, but they arr AI generated, not very structured.  

## Some useful testing data

My earthquakes data and those intermediates (like chunks, chroma-db, ...) are provided in this repository directly. In case it cannot be git pushed, you can checkout the `data_earthquakes_bak.zip` directly, it contains allmy intermediate data and initial documents for your easy start.  
