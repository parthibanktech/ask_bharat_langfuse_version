"""
Answer Generation
-----------------
Simple function to generate answers from retrieved documents.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Use Langfuse OpenAI wrapper - simple as per reference
# instead of: import openai
from langfuse.openai import openai


def generate_answer(query, documents):
    """
    Generate answer using OpenAI.
    
    Args:
        query: User's question
        documents: List of retrieved document texts
        
    Returns:
        Generated answer string
    """
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OPENAI_API_KEY not set. Cannot generate answer. Showing retrieved documents only."
    
    # Combine documents into context
    context = "\n\n".join([f"Document {i+1}:\n{doc[:500]}" for i, doc in enumerate(documents[:3])])
    
    # Create prompt
    prompt = f"""Based on the following documents, answer the question concisely.

Question: {query}

Documents:
{context}

Answer the question based only on the information provided in the documents above. If the answer is not in the documents, say so."""
    
    try:
        response = openai.chat.completions.create(
            name="rag_answer_generation",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            metadata={"task": "rag_answer_generation"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"⚠️ Error generating answer: {str(e)}"

