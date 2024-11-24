from config import client
from routers.vectorstore import get_context

def ask_ai(query: str):
    context = get_context(query)
    chat_completion = client.chat.completions.create(
        messages=[

            {
                "role": "system",
                "content": "you are a helpful assistant. Given a user query and a context, answer the human as clearly as possible."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": f"Context: {context}, query: {query}"
            }
        ],

        # The language model which will generate the completion.
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Print the completion returned by the LLM.
    return chat_completion.choices[0].message.content