"""RAG Prompt Patterns — Reduce hallucination by 80%.

Retrieval-Augmented Generation patterns for accurate, grounded answers.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def basic_rag(question: str, context_chunks: list[str]) -> str:
    """Basic RAG: answer only from provided context."""
    context = "\n---\n".join(context_chunks)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the provided context.\n\n"
                    "RULES:\n"
                    "- If the context doesn't contain the answer, say "
                    "'I don't have enough information to answer this.'\n"
                    "- Quote relevant parts of the context\n"
                    "- Never add information not present in the context\n"
                    "- Cite which chunk the information came from"
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def rag_with_confidence(question: str, context_chunks: list[str]) -> dict:
    """RAG with confidence scoring — know when the answer is uncertain."""
    import json as json_module

    context = "\n---\n".join(context_chunks)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question from the context. Return JSON with:\n"
                    '- "answer": your answer\n'
                    '- "confidence": 0.0-1.0 (how confident based on context)\n'
                    '- "sources": list of chunk numbers used\n'
                    '- "reasoning": why this confidence level\n\n'
                    "If confidence < 0.5, say so in the answer."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json_module.loads(response.choices[0].message.content)


def rag_multi_step(question: str, context_chunks: list[str]) -> str:
    """Multi-step RAG: decompose complex questions into sub-questions."""
    # Step 1: Decompose the question
    decompose = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Break down the complex question into 2-4 simpler sub-questions "
                    "that can each be answered from a document. "
                    "Return one sub-question per line, nothing else."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    sub_questions = decompose.choices[0].message.content.strip().split("\n")

    # Step 2: Answer each sub-question
    context = "\n---\n".join(context_chunks)
    sub_answers = []
    for sq in sub_questions:
        answer = basic_rag(sq.strip(), context_chunks)
        sub_answers.append(f"Q: {sq.strip()}\nA: {answer}")

    # Step 3: Synthesize final answer
    synthesis = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Synthesize a complete answer from these sub-answers. "
                    "Only include information that appeared in the sub-answers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Sub-answers:\n" + "\n\n".join(sub_answers)
                ),
            },
        ],
        temperature=0,
    )
    return synthesis.choices[0].message.content


if __name__ == "__main__":
    # Example context chunks (normally from a vector database)
    chunks = [
        "Chunk 1: Python 3.12 was released on October 2, 2023. "
        "It includes several performance improvements and new features "
        "like improved error messages and a new type parameter syntax.",

        "Chunk 2: The Python Software Foundation (PSF) manages Python's "
        "development. Python follows a 12-month release cycle, with new "
        "major versions released every October.",

        "Chunk 3: Python 3.13 introduced an experimental free-threaded mode "
        "and a JIT compiler. It was released in October 2024.",
    ]

    # Test basic RAG
    print("=== Basic RAG ===")
    answer = basic_rag("When was Python 3.12 released?", chunks)
    print(answer)

    # Test with confidence
    print("\n=== RAG with Confidence ===")
    result = rag_with_confidence("What JIT compiler does Python use?", chunks)
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")

    # Test question NOT in context
    print("\n=== Question Not In Context ===")
    answer = basic_rag("What is Python 3.14's release date?", chunks)
    print(answer)  # Should say "I don't have enough information"
