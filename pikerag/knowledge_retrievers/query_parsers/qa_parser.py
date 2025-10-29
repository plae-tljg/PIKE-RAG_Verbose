# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from pikerag.workflows.common import BaseQaData, MultipleChoiceQaData


def question_as_query(qa: BaseQaData) -> List[str]:
    return [qa.question]


def meta_as_query(qa: BaseQaData, meta_name: str) -> List[str]:
    meta_value = qa.metadata[meta_name]
    if isinstance(meta_value, list):
        return meta_value
    else:
        return [meta_value]


def question_plus_options_as_query(qa: MultipleChoiceQaData) -> List[str]:
    return ["\n".join([qa.question] + list(qa.options.values()))]


def question_plus_each_option_as_query(qa: MultipleChoiceQaData) -> List[str]:
    return [f"{qa.question}\n{option}" for option in qa.options.values()]


def question_and_each_option_as_query(qa: MultipleChoiceQaData) -> List[str]:
    return [qa.question] + list(qa.options.values())
