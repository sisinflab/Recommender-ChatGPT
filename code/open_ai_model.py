# import the OpenAI Python library for calling the OpenAI API
import openai

class OpenAI:
    def __init__(self, model, api_key):
        self.model = model
        openai.api_key = api_key

    def request(self, message):
        # Reference:
        # https://platform.openai.com/docs/api-reference/chat/create
        #
        response = openai.ChatCompletion.create(
            model=self.model, # gpt-3.5-turbo / text-davinci-003
            messages=[
                # experiment 1
                #{"role": "system", "content": "Given a user, as a Recommender System, please provide the top 50 recommendations."},
                # experiment 2 and 3
                {"role": "system", "content": "Given a user, act like a Recommender System."},
                {"role": "user", "content": message}
            ],
            temperature=0,
            max_tokens=750,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response


    def request_davinci(self, message):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=message,
            max_tokens=900,
            temperature=0
        )
        return response