# langchain-critique

This package contains the LangChain integration with Critique

## Installation

```bash
pip install -U langchain-critique
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatCritique` class exposes chat models from Critique.

```python
from langchain_critique import ChatCritique

llm = ChatCritique()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`CritiqueEmbeddings` class exposes embeddings from Critique.

```python
from langchain_critique import CritiqueEmbeddings

embeddings = CritiqueEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`CritiqueLLM` class exposes LLMs from Critique.

```python
from langchain_critique import CritiqueLLM

llm = CritiqueLLM()
llm.invoke("The meaning of life is")
```
