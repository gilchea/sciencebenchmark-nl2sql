# A Small Language Models Based Approach using Prompt Engineering and In-Context Learning for Natural Language to SQL

![Made with Python](https://img.shields.io/badge/Made%20with-Python-brightgreen) ![Maintained](https://img.shields.io/badge/Maintained%3F-yes-yellow) ![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red) ![Made with Pytorch](https://img.shields.io/badge/Made%20with-Transformers-orange)

## Abstract

**Abstract**—The transformation of Natural Language queries into SQL
commands (NL-to-SQL) is a critical task for democratizing data access.
While Large Language Models (LLMs) have shown promising results,
their deployment is often hindered by high computational costs and re-
source demands. This paper explores the potential of high-performance
Small Language Models (SLMs) as efficient alternatives. We propose a
reproducible framework utilizing the 4-bit quantized microsoft/Phi-3-
mini-4k-instruct model for the NL-to-SQL task on the complex CORDIS
scientific dataset. Our approach leverages In-Context Learning (ICL)
augmented by Semantic Retrieval (using BAAI/bge-small and FAISS)
to dynamically select relevant few-shot examples without the need for
fine-tuning. Experimental results demonstrate that while text-based Ex-
act Match (EM) scores remain low due to SQL syntax variations, the
system achieves a competitive Execution Accuracy (EX) of 41% with
3-shot learning. This performance surpasses several baselines, includ-
ing T5-Base and SmBoP, proving that SLMs combined with advanced
prompt engineering can effectively handle complex database schema logic
on a single consumer-grade GPU. We share our experimental setup of
this work on Github repository.
## Paper

(Link paper will be deployed)

## Citation

If you would like to cite this paper, please use the following reference:

```bibtex
@inproceedings{trang2025slmaproach,
  title={A Small Language Models Based Approach using Prompt Engineering and In-Context Learning for Natural Language to SQL},
  author={Nguyen, Kieu-Trang and Le, Quang-Hung},
  year={2025}
}
