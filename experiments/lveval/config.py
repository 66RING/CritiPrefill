from metrics import (
    qa_f1_score_with_gold_ans,
)

DATASET_MAXGEN = {
    "loogle_SD_mixup": 64,
    "loogle_CR_mixup": 64,
    "loogle_MIR_mixup": 64,
    "multifieldqa_en_mixup": 64,
}

DATASET_PROMPT = {
    "loogle_SD_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "loogle_CR_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "loogle_MIR_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
}

DATASET_METRIC = {
    "loogle_SD_mixup": qa_f1_score_with_gold_ans,
    "loogle_CR_mixup": qa_f1_score_with_gold_ans,
    "loogle_MIR_mixup": qa_f1_score_with_gold_ans,
    "multifieldqa_en_mixup": qa_f1_score_with_gold_ans,
}

DATASET_SELECTED = [
    "loogle_MIR_mixup",
    "loogle_CR_mixup",
    "loogle_SD_mixup",
    "multifieldqa_en_mixup",
]

DATASET_LENGTH_LEVEL = [
    '128k',
    '64k',
]
