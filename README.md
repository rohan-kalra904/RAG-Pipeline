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
```
User Prompt: "What are the early warning signs of lung cancer?"

LLM Response:

Early warning signs of lung cancer can be subtle and easily overlooked. Some common symptoms include:

Persistent cough that doesn't go away or worsens over time

Coughing up blood or rust-colored sputum

Chest pain that worsens with deep breathing, coughing, or laughing

Hoarseness or changes in voice

Shortness of breath or wheezing

Recurrent respiratory infections like bronchitis or pneumonia

Unexplained weight loss and loss of appetite

It's important to note that these symptoms can also be caused by other conditions. If you experience any of these signs persistently, especially if you have risk factors such as smoking, it's crucial to consult a healthcare professional for proper evaluation1.

References:

American Cancer Society. "Signs and Symptoms of Lung Cancer." 2024.

National Cancer Institute. "Lung Cancer - Patient Version." 2025.

Follow-up Questions:

What are the main risk factors for developing lung cancer?

How is lung cancer typically diagnosed and staged?

What are the current treatment options available for lung cancer patients?

Disclaimer: This information is provided for educational purposes only and should not be considered as professional medical advice. Always consult with a qualified healthcare provider for proper diagnosis and treatment of any medical condition.
```

## Setup and How to run the code
I have uploaded the code and the given the link to my google colab notebook. The PDFs are stored on my google drive and the API keys used are my own. Rest all dependencies are included in the code as download statements.
So, to run the code just run the entire colab notebook at the last there will be a prompt to enter the query.

![alt text](https://github.com/rohan-kalra904/RAG-Pipeline/blob/79f423ef08c7859d9210582bb798c8c15960139e/Architecture%20Diagram?raw=true)

