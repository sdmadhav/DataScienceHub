from groq import Groq
import os
def load_document(file_path):
    """
    Load text from a file and return its content.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Entire file content as a single string.
    """
    # TODO: Open the file and read its content
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    


def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split the text into chunks of specified size with overlap.

    Args:
        text (str): The full document text.
        chunk_size (int): Maximum number of words per chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        list: List of text chunks (each as a string).
    """
    # TODO: Split text into words
    data = text.split()
    # TODO: Generate overlapping chunks of size 'chunk_size'
    chunks = []
    start, end = 0, chunk_size
    while end < len(data):
        chunk = data[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap
        end = start + chunk_size 

    # TODO: Append each chunk to a list
    return chunks


def score_chunk(chunk, keywords):
    """
    Compute a score for a chunk based on overlapping keywords.

    Args:
        chunk (str): A text chunk.
        keywords (set): Set of keywords from the question.

    Returns:
        int: Number of overlapping words.
    """
    # TODO: Convert chunk to lowercase words
    chunk = chunk.lower()
    return len(set(chunk.split()).intersection(keywords))
    # TODO: Turn into a set and count overlap with keywords
    



def get_best_chunk(chunks, question):
    """
    Select the chunk that best matches the question.

    Args:
        chunks (list): List of text chunks.
        question (str): User's input question.

    Returns:
        str: The chunk with the highest keyword overlap.
    """
    # TODO: Break question into lowercase keywords
    keywords = question.lower().split()
    # TODO: Score each chunk using score_chunk
    maxx = 0
    ans = ''
    for chunk in chunks:
        score = score_chunk(chunk, keywords)
        if score > maxx:
            maxx = score
            ans = chunk

    # TODO: Return the chunk with the highest score
    return ans


def build_prompt(context, question):
    """
    Format the prompt with context and the question.

    Args:
        context (str): Best-matching text chunk.
        question (str): User's input question.

    Returns:
        str: Prompt string formatted for the LLM.
    """
    # The prompt must look like this:
    # Context:
    # {context}
    #
    # Question: {question}
    # Answer:
    forma = f'''Context:
        {context}

        Question: {question}
        Answer:
    '''
    # TODO: Return formatted string
    return forma


def get_answer_from_groq(prompt, api_key):
    """
    Send the prompt to the Groq API and return the model's answer.

    Args:
        prompt (str): The input prompt containing context + question.
        api_key (str): Groq API key provided by the user.

    Returns:
        str: Model-generated answer.
    """
    # TODO: Initialize Groq client

    # TODO: Call chat.completions.create() with:
    #   - model="llama-3.3-70b-versatile"
    #   - messages=[{"role": "user", "content": prompt}]
    #   - temperature=0.3
    #   - max_completion_tokens=512

    # TODO: Return the model's response text

    client = Groq(
        api_key=api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature = 0.3,
        max_completion_tokens = 512,
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    print("Loading document...")
    doc_content = load_document("document.txt")

    print("Splitting into chunks...")
    chunks = chunk_text(doc_content)

    # Ask user question
    question = input("Enter your question: ").lower()

    # Find best chunk
    best_chunk = get_best_chunk(chunks, question)

    # Build prompt
    prompt = build_prompt(best_chunk, question)

    # Call Groq API
    api_key = os.environ.get("GROQ_API_KEY")
    answer = get_answer_from_groq(prompt, api_key)

    print("\nAnswer:")
    print(answer.strip())