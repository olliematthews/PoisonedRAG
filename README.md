# Protecting against knowledge poisoning attacks 

This repo is an investigation of how we can defend against knowledge poisoning attacks, as described in the ["PoisonedRAG" paper](https://arxiv.org/abs/2402.07867).



## Introduction

RAG systems can be a useful way of increasing the useful information output of LLMs by giving them access to information not included in their training data. However, giving RAG systems direct access to potentially untrusted data can open up new vulnerabilities. 

In the [PoisonedRag paper](https://arxiv.org/abs/2402.07867), they show that someone with access to the RAG corpus can inject texts which will be picked up by the retriever. They then show that these texts can make the model output an incorrect answer to a question. It has [also been shown that prompt injections](https://arxiv.org/abs/2302.12173) can be indirectly included via data retrieval. 

In this project, I investigate different ways to mitigate against these attacks. My goal is to attempt to reduce the number of poisoned answers to questions in the nq dataset. I do this through a number of mitigating techniques, listed below:

## Mitigations

### Prompt Tuning




## Acknowledgement
* The code started out as a fork of the [PoisonedRag repo](https://github.com/sleeepeer/PoisonedRAG)


