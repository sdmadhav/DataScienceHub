from groq import Groq

def bullet_point_summary(client, text, num_points=5):
    """
    Summarize text into concise bullet points.

    Args:
        client (Groq): Groq client initialized with API key.
        text (str): Input text to summarize.
        num_points (int): Number of bullet points for the summary.

    Returns:
        str: Generated bullet-point style summary.
    """
    # Build prompt
    prompt = f"Summarize the following text in {num_points} concise bullet points:\n\n{text}"

    # TODO: Call Groq API with model = "llama-3.1-8b-instant"
    # Use temperature = 0.3 and max_completion_tokens = 300
    # Messages:
    # - system message: "You are a concise and clear summarizer."
    # - user message: Should contain the prompt

    # TODO: Parse and return the response text

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a concise and clear summarizer."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature = 0.3,
        max_completion_tokens = 300,
        model="llama-3.1-8b-instant",
    )

    return chat_completion.choices[0].message.content

def abstract_style_summary(client, text, sentence_count=5):
    """
    Summarize text in an academic abstract style.

    Args:
        client (Groq): Groq client initialized with API key.
        text (str): Input text to summarize.
        sentence_count (int): Number of sentences in the abstract.

    Returns:
        str: Generated abstract-style summary.
    """
    # Build prompt
    prompt = f"Summarize the following text as a {sentence_count}-sentence abstract:\n\n{text}"

    # TODO: Call Groq API with model = "llama-3.1-8b-instant"
    # Use temperature = 0.3 and max_completion_tokens = 300
    # Messages:
    # - system message: "You are a concise and clear summarizer."
    # - user message: Should contain the prompt

    # TODO: Parse and return the response text
    chat_completions = client.chat.completions.create(
        messages = [
            {
                "role":"system",
                "content": "You are a concise and clear summarizer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = 0.3,
        max_completion_tokens = 300,
        model = "llama-3.1-8b-instant"
    )


    return chat_completions.choices[0].message.content

def simple_english_summary(client, text, sentence_count=5):
    """
    Summarize text in simple English for a younger audience.

    Args:
        client (Groq): Groq client initialized with API key.
        text (str): Input text to summarize.
        sentence_count (int): Number of sentences in the summary.

    Returns:
        str: Generated simple-English style summary.
    """
    # Build prompt
    prompt = (
        f"Summarize the following text in simple English suitable for a 12-year-old, "
        f"in {sentence_count} sentences:\n\n{text}"
    )

    # TODO: Call Groq API with model = "llama-3.1-8b-instant"
    # Use temperature = 0.3 and max_completion_tokens = 300
    # Messages:
    # - system message: "You are a kind teacher explaining things simply."
    # - user message: Should contain the prompt

    # TODO: Parse and return the response text
    chat_completions = client.chat.completions.create(
        messages = [
            {
                "role" : "system",
                "content": "You are a kind teacher explaining things simply."
            },
            {
                "role" : "user",
                "content" : prompt
            }
        ],
        temperature = 0.3 ,
        max_completion_tokens = 300,
        model= "llama-3.1-8b-instant"
    )
    return chat_completions.choices[0].message.content

# Keyword extractor function
def extract_keywords(text):
    """
    Extract keywords from the given text.

    Args:
        text (str): Input text.

    Returns:
        set: Set of extracted keywords.
    """

    # TODO: Split text into words
    words = text.split()
    # TODO: Convert words to lowercase
    words = [ i.lower() for i in words]
    # TODO: Strip punctuation (.,!?) at the start or end of words.
    new_words = []
    for word in words:
        if word[0] in ['.', ',', '!', '?']:
            word = word[1:]
        if word[-1] in ['.', ',', '!', '?']:
            word = word[:-1]
        new_words.append(word)

    # TODO: Ignore words with length <= 4
    new_words = [word for word in new_words if len(word)>4]
    # TODO: Collect results into a set and return
    
    return set(new_words)

# Choose best summary (Keyword Overlap)
def best_summary_by_keywords(article, summaries) -> str:
    """
    Choose the best summary by measuring keyword overlap with the article.

    Steps:
    - Extract keywords from the article.
    - For each summary:
        * Extract keywords.
        * Compute overlap score using the formula:

          score = overlap_count / (total_article_keywords + 1)

        * Print the score for each summary in the format:
          print(f"Keyword overlap score for {label}: {score:.4f}")

    - Track the summary with the highest score:
        if score > best_score:
            best_label, best_summary, best_score = label, summary, score
    - Return the best summary label and content.

    Args:
        article (str): The original article text.
        summaries (dict): Dictionary of summaries with labels as keys.

    Returns:
        str: The best summary (label + text).
    """
    # TODO: Extract keywords from article
    keywords = extract_keywords(article)
    best_label, best_summary, best_score = "", "", 0
    # TODO: Iterate over summaries
    for label, summary in summaries.items():

        #    - Extract keywords for each summary
        summary_keywords = extract_keywords(summary)
        #    - Compute overlap with article keywords
        count = 0
        for word in summary_keywords:
            if word in keywords:
                count+=1
        
        #    - Calculate score using formula
        score = count / (len(keywords) + 1)
        #    - Print score in required format
        print(f"Keyword overlap score for {label}: {score:.4f}")
        #    - Update best summary if score is higher
        if score > best_score:
            best_label, best_summary, best_score = label, summary, score
    # TODO: Return the best summary with label and text like:
    #       f"Best Summary (by keywords: {best_label}):\n{best_summary}"

    return f"Best Summary (by keywords: {best_label}):\n{best_summary}"

if __name__ == "__main__":
    api_key = ""
    client = Groq(api_key=api_key)

    filepath = "article.txt"
    with open(filepath, "r") as f:
        content = f.read()

    bullet_summary = bullet_point_summary(client, content, num_points=5)
    abstract_summary = abstract_style_summary(client, content, sentence_count=5)
    simple_summary = simple_english_summary(client, content, sentence_count=5)

    print("\n--- Bullet-point Summary ---\n", bullet_summary)
    print("\n--- Abstract Summary ---\n", abstract_summary)
    print("\n--- Simple English Summary ---\n", simple_summary)

    summaries = {
        "Bullet Points": bullet_summary,
        "Abstract": abstract_summary,
        "Simple English": simple_summary,
    }

    final_summary = best_summary_by_keywords(content, summaries)
    print("\nFinal Chosen Summary:\n", final_summary)

