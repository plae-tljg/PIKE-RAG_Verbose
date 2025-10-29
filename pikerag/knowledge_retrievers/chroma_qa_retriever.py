# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
from functools import partial
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from pikerag.knowledge_retrievers.base_qa_retriever import BaseQaRetriever
from pikerag.knowledge_retrievers.mixins.chroma_mixin import ChromaMetaType, ChromaMixin, load_vector_store
from pikerag.utils.config_loader import load_callable, load_embedding_func
from pikerag.utils.logger import Logger
from pikerag.workflows.common import BaseQaData


def load_vector_store_from_configs(
    vector_store_config: dict, embedding_config: dict, collection_name: str=None, persist_directory: str=None,
) -> Chroma:
    if collection_name is None:
        collection_name = vector_store_config["collection_name"]

    if persist_directory is None:
        persist_directory = vector_store_config["persist_directory"]

    embedding = load_embedding_func(
        module_path=embedding_config.get("module_path", None),
        class_name=embedding_config.get("class_name", None),
        **embedding_config.get("args", {}),
    )

    loading_configs: dict = vector_store_config["id_document_loading"]
    ids, documents = load_callable(
        module_path=loading_configs["module_path"],
        name=loading_configs["func_name"],
    )(**loading_configs.get("args", {}))

    exist_ok = vector_store_config.get("exist_ok", True)

    vector_store = load_vector_store(collection_name, persist_directory, embedding, documents, ids, exist_ok)
    return vector_store


class QaChunkRetriever(BaseQaRetriever, ChromaMixin):
    name: str = "QaChunkRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        self._init_query_parser()

        self._load_vector_store()

        self._init_chroma_mixin()

        self.logger = Logger(name=self.name, dump_mode="w", dump_folder=self._log_dir)

    def _init_query_parser(self) -> None:
        query_parser_config: dict = self._retriever_config.get("retrieval_query", None)

        if query_parser_config is None:
            self._main_logger.info(
                msg="`retrieval_query` not configured, default to question_as_query()",
                tag=self.name,
            )

            from pikerag.knowledge_retrievers.query_parsers import question_as_query

            self._query_parser = question_as_query

        else:
            parser_func = load_callable(
                module_path=query_parser_config["module_path"],
                name=query_parser_config["func_name"],
            )
            self._query_parser = partial(parser_func, **query_parser_config.get("args", {}))

    def _load_vector_store(self) -> None:
        assert "vector_store" in self._retriever_config, "vector_store must be defined in retriever part!"
        vector_store_config = self._retriever_config["vector_store"]

        collection_name = vector_store_config.get("collection_name", self.name)
        persist_directory = vector_store_config.get("persist_directory", self._log_dir)
        
        print(f"\n[VECTOR_STORE_LOAD] Configuration:")
        print(f"  Collection name: {collection_name}")
        print(f"  Persist directory: {persist_directory}")
        print(f"  Absolute persist path: {os.path.abspath(persist_directory)}")
        print(f"  Directory exists: {os.path.exists(persist_directory)}")
        
        # Check for existing ChromaDB
        chroma_db_path = os.path.join(persist_directory, "chroma.sqlite3")
        print(f"  ChromaDB file: {chroma_db_path}")
        print(f"  ChromaDB exists: {os.path.exists(chroma_db_path)}")
        
        if os.path.exists(persist_directory):
            files = os.listdir(persist_directory)
            print(f"  Files in persist dir: {files}")
        
        self.vector_store: Chroma = load_vector_store_from_configs(
            vector_store_config=vector_store_config,
            embedding_config=vector_store_config.get("embedding_setting", {}),
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        
        # Check collection count after load
        count = self.vector_store._collection.count()
        print(f"  Vector store loaded. Collection '{collection_name}' has {count} documents")
        return

    def _get_relevant_strings(self, doc_infos: List[Tuple[Document, float]], retrieve_id: str="") -> List[str]:
        contents = [doc.page_content for doc, _ in doc_infos]
        return contents

    def _get_doc_and_score_with_query(self, query: str, retrieve_id: str="", **kwargs) -> List[Tuple[Document, float]]:
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k)
        retrieve_score_threshold = kwargs.get("retrieve_score_threshold", self.retrieve_score_threshold)
        
        print(f"\n[RETRIEVER] Query: {query}")
        print(f"[RETRIEVER] Searching for top-{retrieve_k} chunks with score threshold: {retrieve_score_threshold}")
        
        results = self._get_doc_with_query(query, self.vector_store, retrieve_k, retrieve_score_threshold)
        
        print(f"[RETRIEVER] Found {len(results)} matching chunks")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  Chunk {i}: Score={score:.4f}, Length={len(doc.page_content)}, Metadata={doc.metadata}")
        
        return results

    def retrieve_contents_by_query(self, query: str, retrieve_id: str="", **kwargs) -> List[str]:
        chunk_infos = self._get_doc_and_score_with_query(query, retrieve_id, **kwargs)
        contents = self._get_relevant_strings(chunk_infos, retrieve_id)
        
        print("[RETRIEVER] Chunk contents (first 150 chars of each):")
        for i, content in enumerate(contents, 1):
            preview = content[:150] + "..." if len(content) > 150 else content
            print(f"  Chunk {i}: {preview}")
        
        return contents

    def retrieve_contents(self, qa: BaseQaData, retrieve_id: str="") -> List[str]:
        print(f"\n[RETRIEVER] Starting retrieval for {retrieve_id}")
        print(f"[RETRIEVER] Question: {qa.question}")
        
        queries: List[str] = self._query_parser(qa)
        print(f"[RETRIEVER] Generated {len(queries)} query/queries:")
        for i, q in enumerate(queries, 1):
            print(f"  Query {i}: {q}")
        
        retrieve_k = math.ceil(self.retrieve_k / len(queries))
        print(f"[RETRIEVER] Will retrieve top-{retrieve_k} chunks per query")

        all_chunks: List[str] = []
        for query_idx, query in enumerate(queries, 1):
            print(f"\n[RETRIEVER] Processing Query {query_idx}/{len(queries)}")
            chunks = self.retrieve_contents_by_query(query, retrieve_id, retrieve_k=retrieve_k)
            all_chunks.extend(chunks)

        if len(all_chunks) > 0:
            self.logger.debug(
                msg=f"{retrieve_id}: {len(all_chunks)} strings returned.",
                tag=self.name,
            )
            print(f"[RETRIEVER] Total {len(all_chunks)} chunks retrieved for {retrieve_id}")
        else:
            print(f"[RETRIEVER] WARNING: No chunks retrieved for {retrieve_id}")
        
        return all_chunks


class QaChunkWithMetaRetriever(QaChunkRetriever):
    name: str = "QaChunkWithMetaRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        assert "meta_name" in self._retriever_config, f"meta_name must be specified to use {self.name}"
        self._meta_name = self._retriever_config["meta_name"]

    def _get_relevant_strings(self, doc_infos: List[Tuple[Document, float]], retrieve_id: str="") -> List[str]:
        meta_value_list: List[ChromaMetaType] = list(set([doc.metadata[self._meta_name] for doc, _ in doc_infos]))
        if len(meta_value_list) == 0:
            return []

        _, chunks, _ = self._get_infos_with_given_meta(
            store=self.vector_store,
            meta_name=self._meta_name,
            meta_value=meta_value_list,
        )

        self.logger.debug(f"  {retrieve_id}: {len(meta_value_list)} {self._meta_name} used")
        return chunks
