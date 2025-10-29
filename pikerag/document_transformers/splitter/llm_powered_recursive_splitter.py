# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Callable, Iterable, List, Tuple

from tqdm import tqdm

from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pikerag.llm_client import BaseLLMClient
from pikerag.prompts import CommunicationProtocol
from pikerag.utils.logger import Logger


class LLMPoweredRecursiveSplitter(TextSplitter):
    NAME = "LLMPoweredRecursiveSplitter"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        first_chunk_summary_protocol: CommunicationProtocol,
        last_chunk_summary_protocol: CommunicationProtocol,
        chunk_resplit_protocol: CommunicationProtocol,
        llm_config: dict={},
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        logger: Logger = None,
        **kwargs,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator, add_start_index, strip_whitespace)

        self._base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
            **kwargs,
        )

        self._llm_client = llm_client
        self._llm_config = llm_config

        self._first_chunk_summary_protocol: CommunicationProtocol = first_chunk_summary_protocol
        self._last_chunk_summary_protocol: CommunicationProtocol = last_chunk_summary_protocol
        self._chunk_resplit_protocol: CommunicationProtocol = chunk_resplit_protocol

        self.logger = logger

    def _get_first_chunk_summary(self, text: str, **kwargs) -> str:
        print("\n" + "=" * 80)
        print("STEP 1: Getting first chunk summary")
        print("=" * 80)
        
        # Get the starting part till the end of the first chunk as the content for summary.
        chunks = self._base_splitter.split_text(text)
        first_chunk_start_pos = text.find(chunks[0])
        text_for_summary = text[:first_chunk_start_pos + len(chunks[0])]
        
        print(f"\n[INPUT TO FIRST CHUNK SUMMARY]\nText for summary (first {len(chunks[0])} chars):")
        print("-" * 80)
        print(text_for_summary[:500] + "..." if len(text_for_summary) > 500 else text_for_summary)
        print("-" * 80)

        # Format the message template.
        messages = self._first_chunk_summary_protocol.process_input(content=text_for_summary, **kwargs)
        
        print(f"\n[MESSAGE TO LLM]\n{messages}")
        print("-" * 80)

        # Call client for summary.
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)
        
        print(f"\n[RESPONSE FROM LLM]\n{response}")
        print("-" * 80)

        # Parse response to get the chunk summary.
        summary = self._first_chunk_summary_protocol.parse_output(content=response, **kwargs)
        
        print(f"\n[PARSED SUMMARY]\n{summary}")
        print("=" * 80)
        
        return summary

    def _resplit_chunk_and_generate_summary(
        self, text: str, chunks: List[str], chunk_summary: str, **kwargs,
    ) -> Tuple[str, str, str, str]:
        print("\n" + "=" * 80)
        print("STEP 2: Resplit chunk and generate summaries")
        print("=" * 80)
        
        assert len(chunks) >= 2, f"When calling this function, input chunks length should be no less than 2!"
        text_to_resplit = text[:len(chunks[0]) + len(chunks[1])]
        
        print(f"\n[INPUT TO RESPLIT]\nText to resplit ({len(text_to_resplit)} chars):")
        print("-" * 80)
        print(text_to_resplit[:500] + "..." if len(text_to_resplit) > 500 else text_to_resplit)
        print("-" * 80)
        print(f"\nCurrent chunk summary:\n{chunk_summary}")
        print("-" * 80)

        # Format the message template.
        kwargs["summary"] = chunk_summary
        messages = self._chunk_resplit_protocol.process_input(content=text_to_resplit, **kwargs)
        
        print(f"\n[MESSAGE TO LLM]\n{messages}")
        print("-" * 80)

        # Call client for summary.
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)
        
        print(f"\n[RESPONSE FROM LLM]\n{response}")
        print("-" * 80)

        # Parse response to get the chunk summary.
        result = self._chunk_resplit_protocol.parse_output(content=response, **kwargs)
        first_chunk, first_chunk_summary, second_chunk_summary, dropped_len = result
        
        print("\n[PARSED RESULT]")
        print(f"- First chunk length: {len(first_chunk)}")
        print(f"- First chunk summary: {first_chunk_summary}")
        print(f"- Second chunk summary: {second_chunk_summary}")
        print(f"- Dropped length: {dropped_len}")
        if len(first_chunk) > 0:
            print(f"- First chunk content (first 200 chars):\n{first_chunk[:200]}...")
        print("=" * 80)
        
        return result

    def _get_last_chunk_summary(self, chunk: str, chunk_summary: str, **kwargs) -> str:
        print("\n" + "=" * 80)
        print("STEP 3: Getting last chunk summary")
        print("=" * 80)
        
        print(f"\n[INPUT TO LAST CHUNK SUMMARY]\nChunk ({len(chunk)} chars):")
        print("-" * 80)
        print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
        print("-" * 80)
        print(f"\nCurrent chunk summary:\n{chunk_summary}")
        print("-" * 80)
        
        # Format the message template.
        kwargs["summary"] = chunk_summary
        messages = self._last_chunk_summary_protocol.process_input(content=chunk, **kwargs)
        
        print(f"\n[MESSAGE TO LLM]\n{messages}")
        print("-" * 80)

        # Call client for summary.
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)
        
        print(f"\n[RESPONSE FROM LLM]\n{response}")
        print("-" * 80)

        # Parse response to get the chunk summary.
        summary = self._last_chunk_summary_protocol.parse_output(content=response, **kwargs)
        
        print(f"\n[PARSED SUMMARY]\n{summary}")
        print("=" * 80)
        
        return summary

    def split_text(self, text: str, metadata: dict) -> List[str]:
        docs = self.create_documents(texts=[text], metadatas=[metadata])
        return [doc.page_content for doc in docs]

    def create_documents(self, texts: List[str], metadatas: List[dict], **kwargs) -> List[Document]:
        if len(texts) != len(metadatas):
            error_message = (
                f"Input texts and metadatas should have same length, "
                f"{len(texts)} texts but {len(metadatas)} metadatas are given."
            )
            if self.logger is not None:
                self.logger.error(error_message, tag=self.NAME)
            raise ValueError(error_message)

        ret_docs: List[Document] = []
        for text, metadata in zip(texts, metadatas):
            ret_docs.extend(self.split_documents([Document(page_content=text, metadata=metadata)], **kwargs))
        return ret_docs

    def split_documents(self, documents: Iterable[Document], **kwargs) -> List[Document]:
        ret_docs: List[Document] = []
        for idx, doc in tqdm(enumerate(documents), desc="Splitting Documents", total=len(documents)):
            print("\n" + "#" * 80)
            print(f"# PROCESSING DOCUMENT {idx + 1}")
            print("#" * 80)
            
            text = doc.page_content
            metadata = doc.metadata

            print(f"\nDocument metadata: {metadata}")
            print(f"Document length: {len(text)} characters")
            print("#" * 80)

            text = text.strip()
            chunk_summary = self._get_first_chunk_summary(text, **metadata)
            chunks = self._base_splitter.split_text(text)
            while True:
                if len(chunks) == 1:
                    # Add document for the last chunk
                    chunk_summary = self._get_last_chunk_summary(chunks[0], chunk_summary, **metadata)
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    ret_docs.append(Document(page_content=chunks[0], metadata=chunk_meta))
                    
                    print(f"\n[FINAL CHUNK #{len(ret_docs)} - LAST CHUNK]")
                    print(f"Length: {len(chunks[0])}")
                    print(f"Summary: {chunk_summary}")
                    print(f"Metadata: {chunk_meta}")
                    print("=" * 80)

                    if self.logger is not None:
                        self.logger.debug(
                            msg=(
                                f"{len(ret_docs)}th chunk added (length: {len(chunks[0])}),"
                                f" the last chunk of current document."
                            ),
                            tag=self.NAME,
                        )

                    break

                else:
                    chunk, chunk_summary, next_summary, dropped_len = self._resplit_chunk_and_generate_summary(
                        text, chunks, chunk_summary, **metadata,
                    )

                    if len(chunk) == 0:
                        if self.logger is not None:
                            self.logger.debug(msg="Skip empty re-split first chunk", tag=self.NAME)

                        chunk_summary = next_summary
                        chunks = [chunks[0] + chunks[1]] + chunks[2:]
                        continue

                    # Add document for the first re-splitted chunk
                    chunk_meta = deepcopy(metadata)
                    chunk_meta.update({"summary": chunk_summary})
                    ret_docs.append(Document(page_content=chunk, metadata=chunk_meta))
                    
                    print(f"\n[FINAL CHUNK #{len(ret_docs)}]")
                    print(f"Length: {len(chunk)}")
                    print(f"Summary: {chunk_summary}")
                    print(f"Metadata: {chunk_meta}")
                    print("-" * 80)

                    if self.logger is not None:
                        self.logger.debug(msg=f"{len(ret_docs)}th chunk added (length: {len(chunk)}).", tag=self.NAME)

                    # Update info for remaining text.
                    text = text[dropped_len:].strip()
                    chunk_summary = next_summary
                    chunks = self._base_splitter.split_text(text)
            
            print("\n" + "#" * 80)
            print(f"# DOCUMENT {idx + 1} PROCESSING COMPLETE")
            print(f"# Total chunks for this document: {len(ret_docs)}")
            print("#" * 80 + "\n")

        return ret_docs
