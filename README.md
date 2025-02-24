# RAG-Pipeline

## Data Sources Used - 
The data sources used include only PDFs from famous medical journals and textbooks and are related mostly to cancer. We can also add the functionality to scrape web pages as a further improvement in the project. Also add the funcionality to dynamically add more documents so that an updated database can be maintained

## Text Preprocessing and creating chunks - 
Currently the only text processing being used is the removal of end of line characters and replacing them with whitespace. But further preprocessing such as the removal of links, image references can be used. Also the document can also be broken down into sentences so that all the chunks created during chunking step contain full sentences.
After that tokenizer was used on the text. After carefully considering various LLMs I decided on using the MedEmbed-base-v0.1 for tokenization as it is specialized for medical uses.
Then the texts were divided into chunks of atmost 400 tokens.
```
def chunk_document(text, metadata, max_chunk_size=400):
    # Tokenize the entire text
    tokens = tokenizer.tokenize(text)
    chunks = []

    for i in range(0, len(tokens), max_chunk_size):
        chunk_tokens = tokens[i:i + max_chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)

        # Create chunk-specific metadata
        chunk_metadata = metadata.copy()  # Preserve original metadata
        chunk_metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": i // max_chunk_size,
            "total_chunks": (len(tokens) - 1) // max_chunk_size + 1
        })

        chunks.append({
            "text": chunk_text,
            "metadata": chunk_metadata
        })

    return chunks
```
Code for the text formatter - 
```
def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()

    # Potentially more text formatting functions can go here
    return cleaned_text
```

## Embedding 
All the data after being stored as chunks of tokens it was necessary to convert them to vectors to effectively store and query them. Each chunk was given a unique ID and contained the vector representation. It also contained the title and author of the source for citation purposes and also the source text of the chunk so that it could be used to create a prompt.
The vector database used is Pinecode because of it's easy API usage.
```
def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()

    # Potentially more text formatting functions can go here
    return cleaned_text
```
## Querying and Prompt Format
We use the cosine similarity metric to extract the top k vectors. Once extracted we have all their information in the metadata the prompt is formatted as follows - 
```
    Context: {context}
    References: {references}
    Question: {query}
    
    Instructions:
    1. Answer the question using only the provided context and generally accepted medical knowledge.
    2. If uncertain, state that you don't have enough information to answer accurately.
    3. Include references to the original sources in your answer.
    4. Suggest 2-3 relevant follow-up questions.
    5. Add a disclaimer that this is not official medical advice.
    
    Answer:
```

## LLM Q&A
We chose ChatGPT-4o-mini as our LLM due to its efficiency and performance. It offers several advantages:
60% cheaper than GPT-3.5 Turbo while outperforming it on the MMLU benchmark
128K context window and can output up to 16,384 tokens
Well-suited for high-volume API calls

It has been integrated in our code as follows - 
```
client = openai.Client()

def generate_response(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    print(response)
    return response.choices[0].message.content.strip()
```

## Follow up questions generation
The follow-up question generation approach remains the same, leveraging ChatGPT-4o-mini's capabilities to extract key medical terms and concepts from the original query and generated answer.
The prompt asks the model to create references, follow up questions and then the response from the LLM is given as user output.



