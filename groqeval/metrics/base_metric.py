class BaseMetric:
    def __init__(self, groq_client):
        self.groq_client = groq_client

    def groq_chat_completion(self, messages, model, temperature=0.5, response_format=None):
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format=response_format
        )
        print(chat_completion.choices[0].message.content)
        return(chat_completion)

    def evaluate(self, prompt):
        raise NotImplementedError("This method should be overridden by subclasses")
