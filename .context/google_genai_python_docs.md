TITLE: Generating Content with Text Input (Python)
DESCRIPTION: This snippet demonstrates how to generate content using a text prompt with the `generate_content` method. It specifies the model and the input text, then prints the generated response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_9

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
)
print(response.text)
```

----------------------------------------

TITLE: Providing a String as Content in Python
DESCRIPTION: This snippet demonstrates how a simple Python string provided as `contents` is automatically converted by the Google GenAI SDK into a `types.UserContent` object. The SDK wraps the string in a `types.Part.from_text` object, which is then encapsulated within a `types.UserContent` instance, setting its role to `user`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_13

LANGUAGE: python
CODE:
```
contents='Why is the sky blue?'
```

LANGUAGE: python
CODE:
```
[
types.UserContent(
    parts=[
    types.Part.from_text(text='Why is the sky blue?')
    ]
)
]
```

----------------------------------------

TITLE: Generating Content with Text Input (Python)
DESCRIPTION: This example demonstrates how to use the `client.models.generate_content` method to generate text responses from a specified model, such as 'gemini-2.0-flash-001'. It takes a simple text query as input and prints the generated text response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_9

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
)
print(response.text)
```

----------------------------------------

TITLE: Importing Google Gen AI Modules
DESCRIPTION: This snippet shows the necessary import statements for using the Google Gen AI SDK. It imports the main `genai` module and the `types` module, which provides parameter types like Pydantic Models.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_1

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types
```

----------------------------------------

TITLE: Streaming Text Content Generation in Python
DESCRIPTION: This snippet shows how to generate text content in a streaming fashion, allowing the model's output to be received in chunks rather than a single large response. This is useful for long responses or real-time display, as it processes each `chunk.text` as it arrives.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_37

LANGUAGE: python
CODE:
```
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Generating Content with System Instruction (Python)
DESCRIPTION: This snippet demonstrates how to generate content using the `gemini-2.0-flash-001` model with a specific `system_instruction` to guide the model's response. It configures `max_output_tokens` and `temperature` for the generation. The response text is then printed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_19

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        max_output_tokens=3,
        temperature=0.3,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Installing Google Gen AI SDK
DESCRIPTION: This snippet demonstrates how to install the Google Gen AI Python SDK using pip, the standard package installer for Python. This is the first step to integrate Google's generative models into your Python applications.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_0

LANGUAGE: sh
CODE:
```
pip install google-genai
```

----------------------------------------

TITLE: Installing Google Gen AI SDK (Shell)
DESCRIPTION: This command installs the `google-genai` Python client library using pip, the Python package installer. It is the foundational step to set up the SDK within your development environment, making the library's functionalities available for use.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_0

LANGUAGE: shell
CODE:
```
pip install google-genai
```

----------------------------------------

TITLE: Initializing Client for Gemini Developer API
DESCRIPTION: This snippet demonstrates how to create a client instance for the Gemini Developer API. It requires an `api_key` to authenticate requests. This client is used to interact with Google's generative models.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_2

LANGUAGE: python
CODE:
```
from google import genai

# Only run this block for Gemini Developer API
client = genai.Client(api_key='GEMINI_API_KEY')
```

----------------------------------------

TITLE: Creating a Client for Vertex AI API (Python)
DESCRIPTION: This snippet demonstrates how to initialize a `genai.Client` for integration with the Vertex AI API. It requires setting `vertexai=True` and specifying your Google Cloud `project` ID and `location` to correctly route requests to the Vertex AI service.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_3

LANGUAGE: python
CODE:
```
from google import genai

# Only run this block for Vertex AI API
client = genai.Client(
    vertexai=True, project='your-project-id', location='us-central1'
)
```

----------------------------------------

TITLE: Streaming Text Content Generation (Synchronous) with Gemini 2.0 Flash in Python
DESCRIPTION: This snippet illustrates how to generate content from the Gemini 2.0 Flash model in a synchronous streaming fashion. It iterates over chunks of the response, printing each `chunk.text` as it arrives, allowing for real-time output.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_37

LANGUAGE: python
CODE:
```
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Generating Content with Detailed Typed Configuration (Python)
DESCRIPTION: This example shows how to use Pydantic types for `GenerateContentConfig` parameters when generating content. It queries the `gemini-2.0-flash-001` model with a text prompt and sets various generation parameters like `temperature`, `top_p`, `top_k`, `candidate_count`, `seed`, `max_output_tokens`, `stop_sequences`, `presence_penalty`, and `frequency_penalty`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_20

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=types.Part.from_text(text='Why is the sky blue?'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=['STOP!'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)
```

----------------------------------------

TITLE: Sending Synchronous Streaming Messages in Python Chat
DESCRIPTION: This example shows how to send a synchronous message to a chat session with streaming enabled. It iterates through the response chunks, printing each piece of text as it arrives, providing a more interactive experience.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_52

LANGUAGE: Python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text)
```

----------------------------------------

TITLE: Sending Synchronous Streaming Chat Messages in Python
DESCRIPTION: This example shows how to send a chat message and receive the response in a synchronous streaming fashion. It creates a chat session and then iterates over chunks returned by `chat.send_message_stream`, printing each part of the model's response as it becomes available.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_52

LANGUAGE: python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')  # end='' is optional, for demo purposes.
```

----------------------------------------

TITLE: Enabling Automatic Function Calling with Python Functions (Python)
DESCRIPTION: This example demonstrates automatic function calling by passing a Python function (`get_current_weather`) directly as a tool. The model can then invoke this function based on the user's prompt, and the function's return value is used to generate the response. It queries the `gemini-2.0-flash-001` model about weather in Boston.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_26

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return 'sunny'


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
    ),
)

print(response.text)
```

----------------------------------------

TITLE: Automatic Python Function Calling with Generate Content - Python
DESCRIPTION: This snippet demonstrates how to enable automatic function calling by passing a Python function directly as a tool to `generate_content`. The model can then automatically call this function based on the user's prompt and respond with the result.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_26

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return 'sunny'


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(tools=[get_current_weather]),
)

print(response.text)
```

----------------------------------------

TITLE: Creating Cached Content in Gemini API (Python)
DESCRIPTION: This snippet creates cached content using `client.caches.create`. It configures the cache with a model, content parts (referencing PDF files by URI), a system instruction, a display name, and a time-to-live (TTL) for efficient content reuse.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_60

LANGUAGE: python
CODE:
```
cached_content = client.caches.create(
    model='gemini-2.0-flash-001',
    config=types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role='user',
                parts=[
                    types.Part.from_uri(
                        file_uri=file_uris[0], mime_type='application/pdf'
                    ),
                    types.Part.from_uri(
                        file_uri=file_uris[1],
                        mime_type='application/pdf',
                    ),
                ],
            )
        ],
        system_instruction='What is the sum of the two pdfs?',
        display_name='test cache',
        ttl='3600s',
    ),
)
```

----------------------------------------

TITLE: Configuring API Key via Environment Variable for Gemini Developer API (Bash)
DESCRIPTION: This bash command sets the `GOOGLE_API_KEY` environment variable. When this variable is configured, the `genai.Client` can automatically pick up the API key for authentication, eliminating the need to explicitly pass it during client initialization for the Gemini Developer API.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_4

LANGUAGE: bash
CODE:
```
export GOOGLE_API_KEY='your-api-key'
```

----------------------------------------

TITLE: Declaring and Passing a Function as a Tool in Python
DESCRIPTION: This snippet demonstrates how to manually declare a function using `types.FunctionDeclaration` and pass it as a `types.Tool` to the `generate_content` method. It shows how to configure the model to recognize and suggest this function for specific queries, resulting in a function call part in the response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_29

LANGUAGE: python
CODE:
```
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA',
            ),
        },
        required=['location'],
    ),
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(tools=[tool]),
)

print(response.function_calls[0])
```

----------------------------------------

TITLE: Invoking Function Calls and Passing Responses to Gemini Model (Python)
DESCRIPTION: This snippet demonstrates how to handle a function call returned by the Gemini model, invoke the corresponding Python function (get_current_weather), and then pass the function's response back to the model for further content generation. It includes error handling for the function invocation.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_30

LANGUAGE: python
CODE:
```
from google.genai import types

user_prompt_content = types.Content(
    role='user',
    parts=[types.Part.from_text(text='What is the weather like in Boston?')],
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content


try:
    function_result = get_current_weather(
        **function_call_part.function_call.args
    )
    function_response = {'result': function_result}
except (
    Exception
) as e:  # instead of raising the exception, you can let the model handle it
    function_response = {'error': str(e)}


function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=function_response,
)
function_response_content = types.Content(
    role='tool', parts=[function_response_part]
)

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[
        user_prompt_content,
        function_call_content,
        function_response_content,
    ],
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)
```

----------------------------------------

TITLE: Configuring Content Generation with System Instructions in Python GenAI
DESCRIPTION: This snippet shows how to use `types.GenerateContentConfig` to control model behavior during content generation. Parameters like `system_instruction`, `max_output_tokens`, and `temperature` can be set to guide the model's response and determinism.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_19

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        max_output_tokens=3,
        temperature=0.3,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Configuring Safety Settings for Content Generation (Python)
DESCRIPTION: This snippet shows how to apply safety settings to content generation requests. It attempts to generate content with a prompt that might trigger a safety violation and configures a `SafetySetting` to block content categorized as `HARM_CATEGORY_HATE_SPEECH` with a `BLOCK_ONLY_HIGH` threshold.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_25

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
            )
        ]
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Creating a Tuning Job in Gemini API (Python)
DESCRIPTION: This snippet initiates a tuning job using `client.tunings.tune`. It specifies the `base_model`, the `training_dataset` (prepared in the previous step), and a `CreateTuningJobConfig` for epoch count and tuned model display name.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_64

LANGUAGE: python
CODE:
```
from google.genai import types

tuning_job = client.tunings.tune(
    base_model=model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=1, tuned_model_display_name='test_dataset_examples model'
    ),
)
print(tuning_job)
```

----------------------------------------

TITLE: Generating Content with a Tuned Model in Gemini API (Python)
DESCRIPTION: This snippet demonstrates how to use a fine-tuned model for content generation. It calls `client.models.generate_content`, specifying the `endpoint` of the `tuned_model` obtained from the tuning job, and prints the response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_67

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='why is the sky blue?',
)

print(response.text)
```

----------------------------------------

TITLE: Initiating a Model Tuning Job
DESCRIPTION: Initiates a model tuning job using a specified base model and training dataset. It configures the tuning process with parameters like epoch count and a display name for the tuned model, then prints the job details.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_63

LANGUAGE: python
CODE:
```
from google.genai import types

    tuning_job = client.tunings.tune(
        base_model=model,
        training_dataset=training_dataset,
        config=types.CreateTuningJobConfig(
            epoch_count=1, tuned_model_display_name='test_dataset_examples model'
        ),
    )
    print(tuning_job)
```

----------------------------------------

TITLE: Setting Vertex AI Environment Variables
DESCRIPTION: This snippet shows how to configure environment variables for using the Gemini API on Vertex AI. It sets `GOOGLE_GENAI_USE_VERTEXAI` to true, and specifies the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_5

LANGUAGE: bash
CODE:
```
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

----------------------------------------

TITLE: Sending Chat Messages (Asynchronous Streaming)
DESCRIPTION: Establishes an asynchronous chat session and sends a message, streaming the response chunks as they become available. Each chunk's text is printed incrementally, suitable for real-time output.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_54

LANGUAGE: python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
async for chunk in await chat.send_message_stream('tell me a story'):
    print(chunk.text, end='') # end='' is optional, for demo purposes.
```

----------------------------------------

TITLE: Invoking a Function Call and Passing Response to Model in Python
DESCRIPTION: This snippet illustrates how to extract function call arguments from a model's response, invoke the corresponding Python function (e.g., `get_current_weather`), and then format the function's result (or error) as a `types.Part.from_function_response`. Finally, it shows how to pass this function response back to the model for continued conversation.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_30

LANGUAGE: python
CODE:
```
from google.genai import types

user_prompt_content = types.Content(
    role='user',
    parts=[types.Part.from_text(text='What is the weather like in Boston?')],
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content


try:
    function_result = get_current_weather(
        **function_call_part.function_call.args
    )
    function_response = {'result': function_result}
except (
    Exception
) as e:  # instead of raising the exception, you can let the model handle it
    function_response = {'error': str(e)}


function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=function_response,
)
function_response_content = types.Content(
    role='tool', parts=[function_response_part]
)

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[
        user_prompt_content,
        function_call_content,
        function_response_content,
    ],
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)
```

----------------------------------------

TITLE: Sending Asynchronous Streaming Message with Gemini 2.0 Flash (Python)
DESCRIPTION: This snippet demonstrates how to send an asynchronous streaming message to the Gemini 2.0 Flash model using the `aio.chats.create` and `send_message_stream` methods. It iterates over the streamed chunks and prints the text content, suitable for real-time conversational interactions.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_54

LANGUAGE: python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
async for chunk in await chat.send_message_stream('tell me a story'):
    print(chunk.text)
```

----------------------------------------

TITLE: Generating Content Using Cached Data
DESCRIPTION: Generates content from a Gemini model, leveraging a previously created cached content entry. The model uses the cached data as context for the generation request, and the response text is printed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_61

LANGUAGE: python
CODE:
```
from google.genai import types

    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents='Summarize the pdfs',
        config=types.GenerateContentConfig(
            cached_content=cached_content.name,
        ),
    )
    print(response.text)
```

----------------------------------------

TITLE: Generating JSON Response with Pydantic Schema (Python)
DESCRIPTION: This example shows how to configure the Gemini model to return a JSON response that conforms to a Pydantic BaseModel schema. It defines a CountryInfo Pydantic model and passes it as response_schema in the GenerateContentConfig to ensure structured output.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_33

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from google.genai import types


class CountryInfo(BaseModel):
    name: str
    population: int
    capital: str
    continent: str
    gdp: int
    official_language: str
    total_area_sq_mi: int


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=CountryInfo,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Generating Content with Pydantic Schema in Python
DESCRIPTION: This snippet demonstrates how to use a Pydantic `BaseModel` to define the expected structure of a JSON response from the `generate_content` method. It ensures the model's output conforms to the `CountryInfo` schema, simplifying data parsing and validation.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_33

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from google.genai import types


class CountryInfo(BaseModel):
    name: str
    population: int
    capital: str
    continent: str
    gdp: int
    official_language: str
    total_area_sq_mi: int


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=CountryInfo,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Configuring Generate Content with Pydantic Types - Python
DESCRIPTION: This snippet demonstrates how to use Pydantic types from `google.genai.types` to configure the `generate_content` method. It shows setting various parameters like `temperature`, `top_p`, `top_k`, `candidate_count`, `seed`, `max_output_tokens`, `stop_sequences`, `presence_penalty`, and `frequency_penalty` for a `gemini-2.0-flash-001` model.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_20

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=types.Part.from_text(text='Why is the sky blue?'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=['STOP!'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)
```

----------------------------------------

TITLE: Embedding Single Content with Text Embedding Model in Python
DESCRIPTION: This snippet shows how to generate a numerical embedding for a single piece of text using the `embed_content` method. Embeddings are vector representations useful for tasks like semantic search or clustering.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_45

LANGUAGE: Python
CODE:
```
response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Generating Content Asynchronously with Streaming in Python
DESCRIPTION: This snippet demonstrates how to generate content from a Gemini model asynchronously with streaming. It iterates over chunks of the response, printing each piece of text as it becomes available, which is efficient for long responses.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_41

LANGUAGE: Python
CODE:
```
async for chunk in await client.aio.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Counting Tokens with Gemini 2.0 Flash in Python
DESCRIPTION: This snippet shows how to use the `count_tokens` method to determine the number of tokens in a given text input for the Gemini 2.0 Flash model. It sends a query and prints the token count response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_42

LANGUAGE: python
CODE:
```
response = client.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Generating Content with Uploaded File (Python)
DESCRIPTION: This snippet illustrates how to generate content using an uploaded file. First, `a11.txt` is uploaded via `client.files.upload`, then its reference is included in the `contents` argument of `generate_content` along with a text prompt to summarize the file.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_11

LANGUAGE: python
CODE:
```
file = client.files.upload(file='a11.txt')
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=['Could you summarize this file?', file]
)
print(response.text)
```

----------------------------------------

TITLE: Embedding Multiple Contents with Configuration in Python
DESCRIPTION: This example illustrates how to embed multiple content strings simultaneously and apply a configuration, such as specifying the `output_dimensionality` for the embeddings. It uses `types.EmbedContentConfig` to customize the embedding process, then prints the combined embedding response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_46

LANGUAGE: python
CODE:
```
from google.genai import types

# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10),
)

print(response)
```

----------------------------------------

TITLE: Providing `types.Content` Instance for `generate_content` (Python)
DESCRIPTION: This code demonstrates how to explicitly construct a `types.Content` object with a specified `role` (e.g., 'user') and `parts` (e.g., text) for the `contents` argument of `generate_content`. This method provides granular control over the input structure sent to the model.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_12

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Content(
    role='user',
    parts=[types.Part.from_text(text='Why is the sky blue?')]
)
```

----------------------------------------

TITLE: Sending Chat Messages (Asynchronous)
DESCRIPTION: Initializes an asynchronous chat session with a specified Gemini model and sends a single message. The response text from the model is then printed. This requires an asynchronous client (`client.aio`).
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_53

LANGUAGE: python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
response = await chat.send_message('tell me a story')
print(response.text)
```

----------------------------------------

TITLE: Sending Asynchronous Non-Streaming Messages in Python Chat
DESCRIPTION: This snippet illustrates sending an asynchronous, non-streaming message within a chat session using the `aio` client. This approach is suitable for applications where non-blocking I/O is preferred.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_53

LANGUAGE: Python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
response = await chat.send_message('tell me a story')
print(response.text)
```

----------------------------------------

TITLE: Embedding Single Content in Python
DESCRIPTION: This snippet demonstrates how to generate embeddings for a single piece of text using the `embed_content` method. It specifies the embedding model and the content to be embedded, then prints the resulting embedding response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_45

LANGUAGE: python
CODE:
```
response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Streaming Image Content Generation from Local File in Python
DESCRIPTION: This snippet illustrates how to stream content generation with an image input from a local file system. It reads the image as bytes and uses `types.Part.from_bytes` to include it in the request, enabling the model to analyze the local image and stream its textual output.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_39

LANGUAGE: python
CODE:
```
from google.genai import types

YOUR_IMAGE_PATH = 'your_image_path'
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
with open(YOUR_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
    ],
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Handling API Errors in Google GenAI Python
DESCRIPTION: This snippet demonstrates how to catch and handle `APIError` exceptions raised by the Google GenAI SDK. It attempts to call `generate_content` with an invalid model name and prints the error code (e.g., 404) and message upon failure, providing robust error handling.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_86

LANGUAGE: python
CODE:
```
from google.genai import errors

try:
  client.models.generate_content(
      model="invalid-model-name",
      contents="What is your name?",
  )
except errors.APIError as e:
  print(e.code) # 404
  print(e.message)
```

----------------------------------------

TITLE: Asynchronous Streaming Content Generation with Gemini 2.0 Flash in Python
DESCRIPTION: This snippet illustrates asynchronous streaming content generation using the `client.aio` interface. It uses `async for` to iterate over chunks of the response from the Gemini 2.0 Flash model, printing each `chunk.text` as it is streamed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_41

LANGUAGE: python
CODE:
```
async for chunk in await client.aio.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Generating Content with Uploaded File (Python)
DESCRIPTION: This snippet demonstrates generating content by first uploading a file using `client.files.upload` and then passing the file object along with a text prompt to `generate_content`. This is specifically for the Gemini Developer API.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_11

LANGUAGE: python
CODE:
```
file = client.files.upload(file='a11.txt')
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=['Could you summarize this file?', file]
)
print(response.text)
```

----------------------------------------

TITLE: Applying Safety Settings to Generate Content - Python
DESCRIPTION: This snippet illustrates how to configure safety settings for the `generate_content` method. It shows how to block content based on a specific harm category (e.g., `HARM_CATEGORY_HATE_SPEECH`) at a defined threshold (`BLOCK_ONLY_HIGH`).
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_25

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
            )
        ]
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Streaming Image Content Generation from Local Bytes (Synchronous) with Gemini 2.0 Flash in Python
DESCRIPTION: This snippet shows how to stream content generation using an image loaded from the local file system. It reads the image into bytes, uses `types.Part.from_bytes` to create an image part, and then streams and prints the model's text response about the image.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_39

LANGUAGE: python
CODE:
```
from google.genai import types

YOUR_IMAGE_PATH = 'your_image_path'
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
with open(YOUR_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
    ],
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Creating a Client for Gemini Developer API (Python)
DESCRIPTION: This code initializes a `genai.Client` instance specifically configured for the Gemini Developer API. It requires providing your `api_key` for authentication. This client is essential for making direct requests to the Gemini API endpoints.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_2

LANGUAGE: python
CODE:
```
from google import genai

# Only run this block for Gemini Developer API
client = genai.Client(api_key='GEMINI_API_KEY')
```

----------------------------------------

TITLE: Creating a Client with Environment Variables (Python)
DESCRIPTION: This snippet shows how to initialize a `genai.Client` without any explicit parameters. The client will automatically attempt to use configuration details (like API keys or Vertex AI settings) from environment variables if they have been previously set.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_6

LANGUAGE: python
CODE:
```
from google import genai

client = genai.Client()
```

----------------------------------------

TITLE: Initializing Client with Environment Variables
DESCRIPTION: This snippet illustrates how to create a `genai.Client` instance when necessary environment variables (like `GOOGLE_API_KEY` or Vertex AI specific ones) are already configured. The client automatically picks up these settings.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_6

LANGUAGE: python
CODE:
```
from google import genai

client = genai.Client()
```

----------------------------------------

TITLE: Structuring Content with types.Content Instance
DESCRIPTION: This snippet shows the canonical way to structure the `contents` argument for `generate_content` using a `types.Content` instance. It explicitly defines the `role` and `parts` of the content, allowing for more complex inputs.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_12

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Content(
  role='user',
  parts=[types.Part.from_text(text='Why is the sky blue?')]
)
```

----------------------------------------

TITLE: Streaming Image Content Generation from GCS in Python
DESCRIPTION: This snippet demonstrates how to stream content generation when the input includes an image stored in Google Cloud Storage. It uses `types.Part.from_uri` to reference the image by its GCS URI, allowing the model to process the image and stream its textual description.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_38

LANGUAGE: python
CODE:
```
from google.genai import types

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_uri(
            file_uri='gs://generativeai-downloads/images/scones.jpg',
            mime_type='image/jpeg',
        ),
    ],
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Counting Tokens with Gemini Model in Python
DESCRIPTION: This example shows how to count the number of tokens in a given text input using the `count_tokens` method. This is useful for managing API costs and ensuring inputs fit within model context windows.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_42

LANGUAGE: Python
CODE:
```
response = client.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Generating Enum JSON Response with Gemini 2.0 Flash in Python
DESCRIPTION: This snippet shows how to receive an enum value as a JSON string from the Gemini 2.0 Flash model. It reuses the `InstrumentEnum` and sets `response_mime_type` to 'application/json' and `response_schema` to the enum class, printing the JSON-formatted model output.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_36

LANGUAGE: python
CODE:
```
class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'application/json',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)
```

----------------------------------------

TITLE: Accessing Function Call Parts from Response (Python)
DESCRIPTION: This snippet demonstrates how to access the `function_calls` attribute from the model's response when automatic function calling is disabled. This attribute will contain a list of `types.FunctionCall` objects, indicating the functions the model intended to call.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_28

LANGUAGE: python
CODE:
```
function_calls: Optional[List[types.FunctionCall]] = response.function_calls
```

----------------------------------------

TITLE: Handling API Errors (Synchronous) - Python
DESCRIPTION: This snippet demonstrates how to catch and handle `APIError` exceptions raised by the Google Generative AI SDK. It attempts to generate content with an invalid model name, then prints the error code and message if an `APIError` occurs.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_86

LANGUAGE: python
CODE:
```
try:
    client.models.generate_content(
        model="invalid-model-name",
        contents="What is your name?",
    )
except errors.APIError as e:
    print(e.code) # 404
    print(e.message)
```

----------------------------------------

TITLE: Asynchronous Non-Streaming Content Generation in Python
DESCRIPTION: This snippet demonstrates how to perform asynchronous content generation using the `client.aio` interface. It allows for non-blocking calls to the `generate_content` method, which is beneficial in applications requiring concurrent operations without waiting for the response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_40

LANGUAGE: python
CODE:
```
response = await client.aio.models.generate_content(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
)

print(response.text)
```

----------------------------------------

TITLE: Embedding Multiple Contents with Configuration in Python
DESCRIPTION: This example illustrates how to embed multiple text inputs simultaneously and configure the output dimensionality of the embeddings. It uses `types.EmbedContentConfig` to specify parameters like `output_dimensionality`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_46

LANGUAGE: Python
CODE:
```
from google.genai import types

# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10),
)

print(response)
```

----------------------------------------

TITLE: Streaming Image Content Generation from GCS URI (Synchronous) with Gemini 2.0 Flash in Python
DESCRIPTION: This snippet demonstrates streaming content generation using an image from Google Cloud Storage. It uses `types.Part.from_uri` to create an image part from a GCS URI and `mime_type`, then streams and prints the model's text response about the image.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_38

LANGUAGE: python
CODE:
```
from google.genai import types

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_uri(
            file_uri='gs://generativeai-downloads/images/scones.jpg',
            mime_type='image/jpeg',
        ),
    ],
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Uploading Local Files to Gemini API
DESCRIPTION: Uploads two local PDF files to the Gemini Developer API using the `client.files.upload` method. The uploaded file objects are then printed, typically displaying their names and URIs.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_56

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file2 = client.files.upload(file='2403.05530.pdf')

print(file1)
print(file2)
```

----------------------------------------

TITLE: Asynchronous Non-Streaming Content Generation with Gemini 2.0 Flash in Python
DESCRIPTION: This snippet demonstrates how to perform asynchronous, non-streaming content generation using the `client.aio` interface. It awaits the `generate_content` call for the Gemini 2.0 Flash model and then prints the complete response text once available.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_40

LANGUAGE: python
CODE:
```
response = await client.aio.models.generate_content(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
)

print(response.text)
```

----------------------------------------

TITLE: Uploading PDF Files to Gemini API (Python)
DESCRIPTION: This Python snippet demonstrates uploading two local PDF files (`2312.11805v3.pdf` and `2403.05530.pdf`) to the Gemini API using `client.files.upload`. It then prints the information of the uploaded files, providing their metadata.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_56

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file2 = client.files.upload(file='2403.05530.pdf')

print(file1)
print(file2)
```

----------------------------------------

TITLE: Asynchronously Counting Tokens in Python
DESCRIPTION: This example shows how to asynchronously count tokens for a given text using the `aio.models.count_tokens` method. It requires an `await` call, indicating it's designed for asynchronous programming contexts. The method takes a model and content, then prints the token count response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_44

LANGUAGE: python
CODE:
```
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Listing Base Models Asynchronously with Pagination - Python
DESCRIPTION: This snippet demonstrates asynchronous pagination when listing base models. It sets a `page_size` and shows how to access elements from the async pager and navigate to the next page using `await async_pager.next_page()`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_24

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Listing Base Models with Paging (Asynchronous Python)
DESCRIPTION: This example demonstrates using an asynchronous pager to list base models, setting a `page_size` of 10. It shows how to access pager properties, retrieve items, and asynchronously advance to the next page using `await async_pager.next_page()`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_24

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Sending Synchronous Non-Streaming Chat Messages in Python
DESCRIPTION: This snippet demonstrates how to initiate a multi-turn chat session and send synchronous, non-streaming messages. It creates a chat object with a specified model, then uses `chat.send_message` multiple times to maintain context across turns, printing the model's text responses.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_51

LANGUAGE: python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
response = chat.send_message('tell me a story')
print(response.text)
response = chat.send_message('summarize the story you told me in 1 sentence')
print(response.text)
```

----------------------------------------

TITLE: Polling Batch Job Status Until Completion (Synchronous) - Python
DESCRIPTION: This snippet demonstrates polling the status of a batch prediction job until it reaches a completed state. It repeatedly retrieves the job's current state using `client.batches.get` and pauses for 30 seconds between checks. The final job object is printed upon completion.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_80

LANGUAGE: python
CODE:
```
completed_states = set(
    [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_PAUSED',
    ]
)

while job.state not in completed_states:
    print(job.state)
    job = client.batches.get(name=job.name)
    time.sleep(30)

job
```

----------------------------------------

TITLE: Waiting for Batch Prediction Job Completion in Google GenAI Python
DESCRIPTION: This snippet illustrates how to poll a batch prediction job's status until it reaches a completed state (succeeded, failed, cancelled, or paused). It repeatedly fetches the job status and waits for 30 seconds between checks to avoid excessive API calls.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_80

LANGUAGE: python
CODE:
```
completed_states = set(
    [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_PAUSED',
    ]
)

while job.state not in completed_states:
    print(job.state)
    job = client.batches.get(name=job.name)
    time.sleep(30)

job
```

----------------------------------------

TITLE: Generating Images with Imagen in Python
DESCRIPTION: This snippet demonstrates how to generate an image from a text prompt using the `generate_images` method with the Imagen model. It allows specifying the number of images, including RAI reasons, and the output MIME type via `types.GenerateImagesConfig`. The generated image is then displayed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_47

LANGUAGE: python
CODE:
```
from google.genai import types

# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()
```

----------------------------------------

TITLE: Preparing Tuning Dataset for Model Fine-tuning
DESCRIPTION: Prepares a training dataset for model fine-tuning, conditionally using a GCS URI for Vertex AI or inline examples for the Gemini Developer API. This setup defines the input and output pairs for the tuning process.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_62

LANGUAGE: python
CODE:
```
from google.genai import types

    if client.vertexai:
        model = 'gemini-2.0-flash-001'
        training_dataset = types.TuningDataset(
            gcs_uri='gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl',
        )
    else:
        model = 'models/gemini-2.0-flash-001'
        training_dataset = types.TuningDataset(
            examples=[
                types.TuningExample(
                    text_input=f'Input text {i}',
                    output=f'Output text {i}',
                )
                for i in range(5)
            ],
        )
```

----------------------------------------

TITLE: Sending Synchronous Non-Streaming Messages in Python Chat
DESCRIPTION: This snippet demonstrates creating a chat session and sending multiple synchronous, non-streaming messages. This allows the model to reflect on previous responses, enabling multi-turn conversations.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_51

LANGUAGE: Python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
response = chat.send_message('tell me a story')
print(response.text)
response = chat.send_message('summarize the story you told me in 1 sentence')
print(response.text)
```

----------------------------------------

TITLE: Counting Tokens Asynchronously with Gemini Model in Python
DESCRIPTION: This example demonstrates the asynchronous version of counting tokens using `aio.models.count_tokens`. This method allows for non-blocking operations, which is beneficial in applications requiring concurrent tasks.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_44

LANGUAGE: Python
CODE:
```
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Polling Tuning Job Status
DESCRIPTION: Continuously polls the status of a tuning job until it is no longer in a pending or running state. It prints the current state and pauses for 10 seconds between checks, useful for monitoring long-running operations.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_65

LANGUAGE: python
CODE:
```
import time

    running_states = set(
        [
            'JOB_STATE_PENDING',
            'JOB_STATE_RUNNING',
        ]
    )

    while tuning_job.state in running_states:
        print(tuning_job.state)
        tuning_job = client.tunings.get(name=tuning_job.name)
        time.sleep(10)
```

----------------------------------------

TITLE: Listing Base Models with Pagination - Python
DESCRIPTION: This snippet demonstrates how to use pagination when listing base models. It sets a `page_size` and shows how to access elements from the pager and navigate to the next page using `pager.next_page()`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_22

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Creating Cached Content with Files
DESCRIPTION: Creates a cached content entry for a Gemini model, incorporating two PDF files as parts of the user's content. It conditionally uses GCS URIs for Vertex AI or previously uploaded file URIs for the Gemini Developer API, along with a system instruction and display name.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_59

LANGUAGE: python
CODE:
```
from google.genai import types

    if client.vertexai:
        file_uris = [
            'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
            'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
        ]
    else:
        file_uris = [file1.uri, file2.uri]

    cached_content = client.caches.create(
        model='gemini-2.0-flash-001',
        config=types.CreateCachedContentConfig(
            contents=[
                types.Content(
                    role='user',
                    parts=[
                        types.Part.from_uri(
                            file_uri=file_uris[0], mime_type='application/pdf'
                        ),
                        types.Part.from_uri(
                            file_uri=file_uris[1],
                            mime_type='application/pdf',
                        ),
                    ],
                )
            ],
            system_instruction='What is the sum of the two pdfs?',
            display_name='test cache',
            ttl='3600s',
        ),
    )
```

----------------------------------------

TITLE: Providing a List of Mixed Non-Function Call Parts as Content in Python
DESCRIPTION: This example demonstrates how the Google GenAI SDK processes a list containing various non-function call parts, such as text and URI parts. The SDK groups these diverse parts into a single `types.UserContent` object, ensuring they are all associated with the `user` role for multi-modal input. It requires importing `types` from `google.genai`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_18

LANGUAGE: python
CODE:
```
from google.genai import types

contents = [
types.Part.from_text('What is this image about?'),
types.Part.from_uri(
    file_uri: 'gs://generativeai-downloads/images/scones.jpg',
    mime_type: 'image/jpeg',
)
]
```

LANGUAGE: python
CODE:
```
[
types.UserContent(
        parts=[
        types.Part.from_text('What is this image about?'),
        types.Part.from_uri(
            file_uri: 'gs://generativeai-downloads/images/scones.jpg',
            mime_type: 'image/jpeg',
        )
        ]
    )
]
```

----------------------------------------

TITLE: Providing List of Mixed Non-Function Call Parts as Content in Python GenAI
DESCRIPTION: This snippet demonstrates providing a list containing various non-function call parts, such as text and a URI. The SDK groups these into a single `types.UserContent` object, representing a multi-part user input.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_18

LANGUAGE: python
CODE:
```
from google.genai import types

contents = [
  types.Part.from_text('What is this image about?'),
  types.Part.from_uri(
    file_uri: 'gs://generativeai-downloads/images/scones.jpg',
    mime_type: 'image/jpeg',
  )
]
```

LANGUAGE: python
CODE:
```
[
  types.UserContent(
    parts=[
      types.Part.from_text('What is this image about?'),
      types.Part.from_uri(
        file_uri: 'gs://generativeai-downloads/images/scones.jpg',
        mime_type: 'image/jpeg',
      )
    ]
  )
]
```

----------------------------------------

TITLE: Generating Images with Imagen Model in Python
DESCRIPTION: This snippet demonstrates how to generate an image from a text prompt using the `generate_images` method. It includes configuration options such as the number of images, inclusion of RAI reasons, and output MIME type.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_47

LANGUAGE: Python
CODE:
```
from google.genai import types

# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()
```

----------------------------------------

TITLE: Generating Content with Raw JSON Schema in Python
DESCRIPTION: This snippet shows how to define a response schema directly using a Python dictionary that represents a JSON schema. This provides an alternative to Pydantic models for specifying the expected structure of the generated content, ensuring the output is a valid JSON object with defined properties and types.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_34

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema={
            'required': [
                'name',
                'population',
                'capital',
                'continent',
                'gdp',
                'official_language',
                'total_area_sq_mi',
            ],
            'properties': {
                'name': {'type': 'STRING'},
                'population': {'type': 'INTEGER'},
                'capital': {'type': 'STRING'},
                'continent': {'type': 'STRING'},
                'gdp': {'type': 'INTEGER'},
                'official_language': {'type': 'STRING'},
                'total_area_sq_mi': {'type': 'INTEGER'},
            },
            'type': 'OBJECT',
        },
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Manually Declaring and Passing a Function as a Tool (Python)
DESCRIPTION: This snippet illustrates how to manually declare a function using `types.FunctionDeclaration` and wrap it in a `types.Tool` object. This allows for explicit control over function definitions when passing them to the model for function calling, rather than relying on automatic introspection of Python functions. The example defines a `get_current_weather` function.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_29

LANGUAGE: python
CODE:
```
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA',
            ),
        },
        required=['location'],
    ),
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
```

----------------------------------------

TITLE: Generating JSON Response with Dictionary Schema (Python)
DESCRIPTION: This snippet begins to illustrate how to configure the Gemini model to return a JSON response using a dictionary-based schema. It sets response_mime_type to application/json and starts defining response_schema as a dictionary.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_34

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema={

```

----------------------------------------

TITLE: Generating Enum Text Response with Gemini 2.0 Flash in Python
DESCRIPTION: This snippet demonstrates how to configure the Gemini 2.0 Flash model to return an enum value as a plain text response. It defines an `InstrumentEnum` and sets `response_mime_type` to 'text/x.enum' and `response_schema` to the enum class, then prints the model's text output.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_35

LANGUAGE: python
CODE:
```
from enum import Enum

class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)
```

----------------------------------------

TITLE: Listing Tuned Models with Pager (Asynchronous) - Python
DESCRIPTION: This asynchronous example shows how to list tuned models using an asynchronous pager object. It retrieves the first page, prints its size and the first item, then asynchronously advances to the next page and prints its first item.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_72

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Listing Tuning Jobs (Asynchronous Iteration) - Python
DESCRIPTION: This asynchronous snippet shows how to iterate through a paginated list of tuning jobs. It uses `client.aio.tunings.list` with a configuration for page size. Each tuning job is then printed asynchronously.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_76

LANGUAGE: python
CODE:
```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)
```

----------------------------------------

TITLE: Providing a Single Non-Function Call Part as Content in Python
DESCRIPTION: This snippet illustrates providing a non-function call part, such as a URI part for an image, using `types.Part.from_uri`. The SDK automatically converts such parts into a `types.UserContent` object, assigning the `user` role, as these typically represent user-provided input like images or files. It requires importing `types` from `google.genai`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_17

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Part.from_uri(
file_uri: 'gs://generativeai-downloads/images/scones.jpg',
mime_type: 'image/jpeg',
)
```

LANGUAGE: python
CODE:
```
[
types.UserContent(parts=[
        types.Part.from_uri(
        file_uri: 'gs://generativeai-downloads/images/scones.jpg',
        mime_type: 'image/jpeg',
        )
    ])
]
```

----------------------------------------

TITLE: Providing a Single Function Call Part as Content in Python
DESCRIPTION: This snippet shows how to provide a single function call part using `types.Part.from_function_call`. The SDK automatically converts this into a `types.ModelContent` object, setting its `role` field to `model`, as function calls typically represent model-generated content. It requires importing `types` from `google.genai`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_15

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Part.from_function_call(
    name='get_weather_by_location',
    args={'location': 'Boston'}
)
```

LANGUAGE: python
CODE:
```
[
types.ModelContent(
    parts=[
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
    )
    ]
)
]
```

----------------------------------------

TITLE: Initializing Client for Vertex AI API
DESCRIPTION: This snippet shows how to create a client instance specifically for the Vertex AI API. It requires setting `vertexai=True`, along with your Google Cloud `project` ID and `location` for proper authentication and resource targeting.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_3

LANGUAGE: python
CODE:
```
from google import genai

# Only run this block for Vertex AI API
client = genai.Client(
    vertexai=True, project='your-project-id', location='us-central1'
)
```

----------------------------------------

TITLE: Listing All Base Models Asynchronously - Python
DESCRIPTION: This snippet shows how to asynchronously iterate through and print all available base models using `client.aio.models.list()`. It requires `await` and `async for` for non-blocking operations.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_23

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list():
    print(job)
```

----------------------------------------

TITLE: Configuring Environment Variables for Vertex AI API (Bash)
DESCRIPTION: These bash commands set the necessary environment variables (`GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) to configure the `genai.Client` for Vertex AI. This allows the client to initialize automatically with the correct project and location settings without explicit parameters.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_5

LANGUAGE: bash
CODE:
```
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

----------------------------------------

TITLE: Listing Batch Prediction Jobs Asynchronously in Google GenAI Python
DESCRIPTION: This asynchronous example iterates through batch prediction jobs using `await client.aio.batches.list`, providing non-blocking retrieval. It uses `types.ListBatchJobsConfig` for pagination, suitable for concurrent operations.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_83

LANGUAGE: python
CODE:
```
async for job in await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
):
    print(job)
```

----------------------------------------

TITLE: Listing Tuning Jobs with Pager (Asynchronous) - Python
DESCRIPTION: This asynchronous example demonstrates listing tuning jobs using an asynchronous pager object. It retrieves the first page, prints its size and the first item, then asynchronously advances to the next page and prints its first item.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_77

LANGUAGE: python
CODE:
```
async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Providing a List of Strings as Content in Python
DESCRIPTION: This example illustrates how the Google GenAI SDK processes a list of strings provided as `contents`. The SDK converts each string into a `types.Part.from_text` object and then combines them into a single `types.UserContent` instance, ensuring all parts are associated with the `user` role.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_14

LANGUAGE: python
CODE:
```
contents=['Why is the sky blue?', 'Why is the cloud white?']
```

LANGUAGE: python
CODE:
```
[
types.UserContent(
    parts=[
    types.Part.from_text(text='Why is the sky blue?'),
    types.Part.from_text(text='Why is the cloud white?'),
    ]
)
]
```

----------------------------------------

TITLE: Setting GOOGLE_API_KEY Environment Variable
DESCRIPTION: This snippet demonstrates how to set the `GOOGLE_API_KEY` environment variable for the Gemini Developer API. This allows the `genai.Client()` to automatically pick up the API key without explicitly passing it in the code.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_4

LANGUAGE: bash
CODE:
```
export GOOGLE_API_KEY='your-api-key'
```

----------------------------------------

TITLE: Listing Batch Jobs with Pager (Synchronous) - Python
DESCRIPTION: This example demonstrates listing batch prediction jobs using a pager object for controlled pagination. It retrieves the first page, prints its size and the first item, then advances to the next page and prints its first item.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_82

LANGUAGE: python
CODE:
```
from google.genai import types

pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Computing Tokens with Vertex AI Python
DESCRIPTION: This snippet demonstrates how to compute tokens for a given text using the `compute_tokens` method. This functionality is specifically supported within Vertex AI. It takes a model identifier and content as input and prints the response.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_43

LANGUAGE: python
CODE:
```
response = client.models.compute_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Providing List of Text Strings as Content in Python GenAI
DESCRIPTION: This snippet shows how a Python list of strings is provided as input for content. The Google GenAI SDK automatically converts this into a single `types.UserContent` object, where each string becomes a `types.Part.from_text` within the `parts` list, representing user-provided text.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_14

LANGUAGE: python
CODE:
```
contents=['Why is the sky blue?', 'Why is the cloud white?']
```

LANGUAGE: python
CODE:
```
[
  types.UserContent(
    parts=[
      types.Part.from_text(text='Why is the sky blue?'),
      types.Part.from_text(text='Why is the cloud white?'),
    ]
  )
]
```

----------------------------------------

TITLE: Computing Tokens with Gemini Model in Python (Vertex AI)
DESCRIPTION: This snippet illustrates how to compute tokens for a text input using the `compute_tokens` method. It's important to note that this specific functionality is currently only supported when using the client with Vertex AI.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_43

LANGUAGE: Python
CODE:
```
response = client.models.compute_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Importing Google Gen AI Modules (Python)
DESCRIPTION: This snippet imports the core `genai` module and the `types` submodule from the `google.genai` package. These imports are crucial for accessing the main client functionalities and for defining specific parameter types when interacting with the Google Generative AI APIs.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_1

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types
```

----------------------------------------

TITLE: Listing All Base Models (Synchronous Python)
DESCRIPTION: This snippet demonstrates how to synchronously iterate and print all available base models using the `client.models.list()` method. It fetches models one by one as they are yielded by the iterator.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_21

LANGUAGE: python
CODE:
```
for model in client.models.list():
    print(model)
```

----------------------------------------

TITLE: Using a Fine-tuned Model for Content Generation
DESCRIPTION: Generates content using the endpoint of a previously fine-tuned model. It sends a prompt to the tuned model and prints the generated response, demonstrating how to interact with a custom-trained model.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_66

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
        model=tuning_job.tuned_model.endpoint,
        contents='why is the sky blue?',
    )

    print(response.text)
```

----------------------------------------

TITLE: Paginating Tuning Jobs Synchronously in Google GenAI Python
DESCRIPTION: This snippet demonstrates how to use the pager object returned by `client.tunings.list` to manage pagination for tuning jobs. It shows accessing the page size, the first element, and advancing to the next page using `pager.next_page()`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_75

LANGUAGE: python
CODE:
```
pager = client.tunings.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Providing List of Function Call Parts as Content in Python GenAI
DESCRIPTION: This snippet illustrates providing a list of `types.Part.from_function_call` objects. The SDK groups these consecutive function call parts into a single `types.ModelContent` object, signifying multiple model-initiated function calls within the conversation.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_16

LANGUAGE: python
CODE:
```
from google.genai import types

contents = [
  types.Part.from_function_call(
    name='get_weather_by_location',
    args={'location': 'Boston'}
  ),
  types.Part.from_function_call(
    name='get_weather_by_location',
    args={'location': 'New York'}
  ),
]
```

LANGUAGE: python
CODE:
```
[
  types.ModelContent(
    parts=[
      types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
      ),
      types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'New York'}
      )
    ]
  )
]
```

----------------------------------------

TITLE: Paginating Tuning Jobs Asynchronously in Google GenAI Python
DESCRIPTION: This asynchronous snippet shows how to use the pager object from `await client.aio.tunings.list` to manage pagination for tuning jobs. It demonstrates accessing page size, elements, and advancing to the next page using `await async_pager.next_page()`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_77

LANGUAGE: python
CODE:
```
async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Listing All Base Models - Python
DESCRIPTION: This snippet iterates through and prints all available base models using the `client.models.list()` method. It provides a simple way to retrieve a list of models without explicit pagination.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_21

LANGUAGE: python
CODE:
```
for model in client.models.list():
    print(model)
```

----------------------------------------

TITLE: Preparing Tuning Dataset for Gemini API (Python)
DESCRIPTION: This snippet prepares the training dataset for a tuning job. It conditionally sets the `model` and `training_dataset` based on whether `client.vertexai` is enabled, supporting GCS URIs for Vertex AI and inline examples for Gemini Developer API.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_63

LANGUAGE: python
CODE:
```
from google.genai import types

if client.vertexai:
    model = 'gemini-2.0-flash-001'
    training_dataset = types.TuningDataset(
        gcs_uri='gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl',
    )
else:
    model = 'models/gemini-2.0-flash-001'
    training_dataset = types.TuningDataset(
        examples=[
            types.TuningExample(
                text_input=f'Input text {i}',
                output=f'Output text {i}',
            )
            for i in range(5)
        ],
    )
```

----------------------------------------

TITLE: Providing Single Function Call Part as Content in Python GenAI
DESCRIPTION: This example demonstrates providing a single `types.Part.from_function_call` as content. The SDK processes this into a `types.ModelContent` object, indicating that this part represents a function call initiated by the model, with the `role` field fixed to `model`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_15

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Part.from_function_call(
  name='get_weather_by_location',
  args={'location': 'Boston'}
)
```

LANGUAGE: python
CODE:
```
[
  types.ModelContent(
    parts=[
      types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
      )
    ]
  )
]
```

----------------------------------------

TITLE: Setting API Version to v1 for Vertex AI Client (Python)
DESCRIPTION: This code configures the `genai.Client` for Vertex AI to use the stable `v1` API endpoints instead of the default beta endpoints. This is achieved by passing `http_options` with `api_version='v1'` during client initialization, ensuring interaction with a stable API version.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_7

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1',
    http_options=types.HttpOptions(api_version='v1')
)
```

----------------------------------------

TITLE: Listing Tuning Jobs Asynchronously in Google GenAI Python
DESCRIPTION: This asynchronous example iterates through tuning jobs using `await client.aio.tunings.list`, providing a non-blocking way to retrieve job lists. It also uses a `page_size` configuration for pagination.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_76

LANGUAGE: python
CODE:
```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)
```

----------------------------------------

TITLE: Generating Enum Text Response in Python
DESCRIPTION: This snippet illustrates how to configure the `generate_content` method to return a single enum value as a plain text response. By setting `response_mime_type` to 'text/x.enum' and providing an `Enum` class, the model is guided to output one of the predefined enum members.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_35

LANGUAGE: python
CODE:
```
class InstrumentEnum(Enum):
  PERCUSSION = 'Percussion'
  STRING = 'String'
  WOODWIND = 'Woodwind'
  BRASS = 'Brass'
  KEYBOARD = 'Keyboard'

response = client.models.generate_content(
      model='gemini-2.0-flash-001',
      contents='What instrument plays multiple notes at once?',
      config={
          'response_mime_type': 'text/x.enum',
          'response_schema': InstrumentEnum,
      },
  )
print(response.text)
```

----------------------------------------

TITLE: Accessing Function Call Parts from Response - Python
DESCRIPTION: This snippet demonstrates how to access the `function_calls` attribute from the `response` object when automatic function calling is disabled. It shows that the response will contain a list of `types.FunctionCall` objects, which can then be processed manually.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_28

LANGUAGE: python
CODE:
```
function_calls: Optional[List[types.FunctionCall]] = response.function_calls
```

----------------------------------------

TITLE: Paginating Tuned Models List in Gemini API (Python)
DESCRIPTION: This snippet demonstrates how to use the pager object for listing tuned models. It retrieves the page size, accesses elements by index, and uses `pager.next_page()` to fetch the next set of results, enabling efficient navigation through large lists.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_70

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10, 'query_base': False})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Limiting Automatic Function Calling Turns with ANY Tool Config in Python
DESCRIPTION: This snippet shows how to configure the maximum number of automatic function call turns when the `tool_config` mode is `ANY`. By setting `maximum_remote_calls` in `types.AutomaticFunctionCallingConfig`, you can control how many times the SDK will automatically invoke functions before returning a function call part to the user.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_32

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=2
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

----------------------------------------

TITLE: Disabling Automatic Function Calling with ANY Tool Config in Python
DESCRIPTION: This snippet demonstrates how to explicitly disable automatic function calling when the `tool_config` mode is set to `ANY`. It shows how to pass a Python function directly as a tool and use `types.AutomaticFunctionCallingConfig(disable=True)` to prevent the SDK from automatically invoking the function.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_31

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

----------------------------------------

TITLE: Disabling Automatic Function Calling (Python)
DESCRIPTION: This snippet shows how to explicitly disable automatic function calling when passing a Python function as a tool. By setting `automatic_function_calling.disable` to `True`, the model will return a function call part in the response instead of automatically invoking the function.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_27

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    ),
)
```

----------------------------------------

TITLE: Listing All Base Models (Asynchronous Python)
DESCRIPTION: This snippet illustrates how to asynchronously iterate and print all available base models using the `client.aio.models.list()` method. It's suitable for asynchronous contexts, fetching models as they become available.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_23

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list():
    print(job)
```

----------------------------------------

TITLE: Disabling Automatic Function Calling in Gemini (Python)
DESCRIPTION: This example shows how to configure the Gemini model to disable automatic function calling when the tool_config mode is set to ANY. It defines a get_current_weather tool and explicitly sets automatic_function_calling.disable to True in the GenerateContentConfig.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_31

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

----------------------------------------

TITLE: Providing Single Non-Function Call Part (URI) as Content in Python GenAI
DESCRIPTION: This example shows how a single non-function call part, specifically a URI pointing to an image, is handled. The SDK converts this into a `types.UserContent` object, indicating that it's a user-provided input, with the `role` field fixed to `user`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_17

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Part.from_uri(
  file_uri: 'gs://generativeai-downloads/images/scones.jpg',
  mime_type: 'image/jpeg',
)
```

LANGUAGE: python
CODE:
```
[
  types.UserContent(parts=[
    types.Part.from_uri(
     file_uri: 'gs://generativeai-downloads/images/scones.jpg',
      mime_type: 'image/jpeg',
    )
  ])
]
```

----------------------------------------

TITLE: Listing Tuning Jobs with Pager (Synchronous) - Python
DESCRIPTION: This example demonstrates listing tuning jobs using a pager object for controlled pagination. It retrieves the first page, prints its size and the first item, then advances to the next page and prints its first item.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_75

LANGUAGE: python
CODE:
```
pager = client.tunings.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Updating a Tuned Model from Pager (Synchronous) - Python
DESCRIPTION: This snippet demonstrates updating a tuned model obtained from a pager object. It retrieves the first model from the `pager`, then updates its display name and description using `client.models.update`. The updated model is then printed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_73

LANGUAGE: python
CODE:
```
model = pager[0]

model = client.models.update(
    model=model.name,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    ),
)

print(model)
```

----------------------------------------

TITLE: Generating Videos with Veo in Python
DESCRIPTION: This example illustrates how to generate a video from a text prompt using the `generate_videos` method with the Veo model. It creates an asynchronous operation, which is then polled until completion. The configuration allows setting the number of videos, frames per second, duration, and prompt enhancement. Video generation support is currently allowlisted.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_50

LANGUAGE: python
CODE:
```
from google.genai import types

# Create operation
operation = client.models.generate_videos(
    model='veo-2.0-generate-001',
    prompt='A neon hologram of a cat driving at top speed',
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        fps=24,
        duration_seconds=5,
        enhance_prompt=True,
    ),
)

# Poll operation
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

video = operation.result.generated_videos[0].video
video.show()
```

----------------------------------------

TITLE: Listing Base Models with Paging (Synchronous Python)
DESCRIPTION: This example shows how to use a synchronous pager to list base models, specifying a `page_size` of 10. It demonstrates accessing the `page_size` property, retrieving the first item, and advancing to the next page using `next_page()`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_22

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Deleting a File from Gemini API (Python)
DESCRIPTION: This snippet demonstrates how to delete an uploaded file from the Gemini API. It first uploads a file for demonstration purposes and then uses `client.files.delete` with the file's `name` to remove it from the service.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_58

LANGUAGE: python
CODE:
```
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)
```

----------------------------------------

TITLE: Paginating Batch Prediction Jobs Asynchronously in Google GenAI Python
DESCRIPTION: This asynchronous snippet demonstrates using the pager object from `await client.aio.batches.list` for batch prediction jobs. It shows accessing page size, elements, and advancing to the next page using `await async_pager.next_page()`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_84

LANGUAGE: python
CODE:
```
async_pager = await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
)
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Generating Videos with Veo Model in Python
DESCRIPTION: This example shows how to generate a video from a text prompt using the `generate_videos` method. It demonstrates creating an operation, polling it for completion, and then displaying the generated video.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_50

LANGUAGE: Python
CODE:
```
from google.genai import types

# Create operation
operation = client.models.generate_videos(
    model='veo-2.0-generate-001',
    prompt='A neon hologram of a cat driving at top speed',
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        fps=24,
        duration_seconds=5,
        enhance_prompt=True,
    ),
)

# Poll operation
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

video = operation.result.generated_videos[0].video
video.show()
```

----------------------------------------

TITLE: Providing a List of Function Call Parts as Content in Python
DESCRIPTION: This example demonstrates how the Google GenAI SDK handles a list of multiple function call parts. The SDK groups these parts into a single `types.ModelContent` object, ensuring that all function calls are correctly attributed to the `model` role within the content structure. It requires importing `types` from `google.genai`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_16

LANGUAGE: python
CODE:
```
from google.genai import types

contents = [
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
    ),
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'New York'}
    ),
]
```

LANGUAGE: python
CODE:
```
[
types.ModelContent(
    parts=[
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
    ),
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'New York'}
    )
    ]
)
]
```

----------------------------------------

TITLE: Paginating Batch Prediction Jobs Synchronously in Google GenAI Python
DESCRIPTION: This snippet shows how to use the pager object from `client.batches.list` to manage pagination for batch prediction jobs. It demonstrates accessing page size, the first element, and advancing to the next page using `pager.next_page()`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_82

LANGUAGE: python
CODE:
```
pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Generating Enum JSON Response in Python
DESCRIPTION: This snippet demonstrates how to receive an enum value as a JSON string. By setting `response_mime_type` to 'application/json' and providing an `Enum` class, the model's output will be one of the enum members, enclosed in double quotes as a JSON string.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_36

LANGUAGE: python
CODE:
```
from enum import Enum

class InstrumentEnum(Enum):
  PERCUSSION = 'Percussion'
  STRING = 'String'
  WOODWIND = 'Woodwind'
  BRASS = 'Brass'
  KEYBOARD = 'Keyboard'

response = client.models.generate_content(
      model='gemini-2.0-flash-001',
      contents='What instrument plays multiple notes at once?',
      config={
          'response_mime_type': 'application/json',
          'response_schema': InstrumentEnum,
      },
  )
print(response.text)
```

----------------------------------------

TITLE: Retrieving a Batch Prediction Job by Name in Google GenAI Python
DESCRIPTION: This example shows how to fetch the details of an existing batch prediction job using its name via `client.batches.get`. It then prints the current state of the retrieved job, which is useful for monitoring progress.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_79

LANGUAGE: python
CODE:
```
# Get a job by name
job = client.batches.get(name=job.name)

job.state
```

----------------------------------------

TITLE: Limiting Automatic Function Calling Turns in Gemini (Python)
DESCRIPTION: This snippet demonstrates how to limit the number of automatic function call turns for the Gemini model when tool_config mode is ANY. It sets automatic_function_calling.maximum_remote_calls to 2 to allow for one turn of automatic function calling (as x+1 turns are configured for x turns).
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_32

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=2
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

----------------------------------------

TITLE: Generating Content with Cached Data in Gemini API (Python)
DESCRIPTION: This snippet shows how to generate content using a pre-existing cache. It calls `client.models.generate_content` with a model, a user prompt, and a `GenerateContentConfig` that references the `cached_content.name`, leveraging cached data for faster responses.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_62

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Summarize the pdfs',
    config=types.GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Listing Tuned Models (Asynchronous Iteration) - Python
DESCRIPTION: This asynchronous snippet demonstrates iterating through a paginated list of tuned models. It uses `client.aio.models.list` with a configuration for page size and to exclude base models. Each model is then printed asynchronously.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_71

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}):
    print(job)
```

----------------------------------------

TITLE: Editing Images with Imagen Model in Python (Vertex AI)
DESCRIPTION: This snippet demonstrates how to edit an image using the `edit_image` method, incorporating raw and mask reference images. It configures the edit mode and output, and is currently only supported in Vertex AI.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_49

LANGUAGE: Python
CODE:
```
# Edit the generated image from above
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage

raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-001',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode='EDIT_MODE_INPAINT_INSERTION',
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response3.generated_images[0].image.show()
```

----------------------------------------

TITLE: Listing Tuned Models with Pager (Synchronous) - Python
DESCRIPTION: This example shows how to list tuned models using a pager object for more granular control over pagination. It retrieves the first page, prints its size and the first item, then advances to the next page and prints its first item.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_70

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10, 'query_base': False})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Asynchronously Paginating Tuned Models List in Gemini API (Python)
DESCRIPTION: This snippet demonstrates asynchronous pagination for listing tuned models. It uses `await client.aio.models.list` to get an async pager, accesses elements, and uses `await async_pager.next_page()` for subsequent pages, ideal for concurrent operations.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_72

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Creating a Batch Prediction Job (Synchronous) - Python
DESCRIPTION: This snippet demonstrates creating a batch prediction job by specifying only the model and source data. The destination and job display name are auto-populated. It uses `client.batches.create` and then prints the created job object.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_78

LANGUAGE: python
CODE:
```
# Specify model and source file only, destination and job display name will be auto-populated
job = client.batches.create(
    model='gemini-2.0-flash-001',
    src='bq://my-project.my-dataset.my-table',
)

job
```

----------------------------------------

TITLE: Asynchronously Listing Tuned Models in Gemini API (Python)
DESCRIPTION: This snippet shows how to asynchronously iterate and print a list of tuned models using `client.aio.models.list`. It's suitable for non-blocking operations in an asynchronous context, improving application responsiveness.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_71

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}):
    print(job)
```

----------------------------------------

TITLE: Retrieving Cached Content from Gemini API (Python)
DESCRIPTION: This snippet demonstrates how to retrieve information about a previously created cached content entry using `client.caches.get`. It requires the `name` of the cached content as a parameter to fetch its details and status.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_61

LANGUAGE: python
CODE:
```
cached_content = client.caches.get(name=cached_content.name)
```

----------------------------------------

TITLE: Listing Tuned Models (Synchronous Iteration) - Python
DESCRIPTION: This snippet demonstrates iterating through a paginated list of tuned models. It uses `client.models.list` with a configuration to specify page size and exclude base models. Each model in the list is then printed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_69

LANGUAGE: python
CODE:
```
for model in client.models.list(config={'page_size': 10, 'query_base': False}):
    print(model)
```

----------------------------------------

TITLE: Listing Tuning Jobs Synchronously in Google GenAI Python
DESCRIPTION: This example shows how to iterate through a list of tuning jobs using the synchronous `client.tunings.list` method. It prints each job found and uses a `page_size` configuration for basic pagination.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_74

LANGUAGE: python
CODE:
```
for job in client.tunings.list(config={'page_size': 10}):
    print(job)
```

----------------------------------------

TITLE: Listing Tuned Models in Gemini API (Python)
DESCRIPTION: This snippet iterates and prints a list of tuned models available in the Gemini API. It uses `client.models.list` with a configuration to specify page size and to exclude base models (`query_base: False`), focusing only on fine-tuned models.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_69

LANGUAGE: python
CODE:
```
for model in client.models.list(config={'page_size': 10, 'query_base': False}):
    print(model)
```

----------------------------------------

TITLE: Setting API Version to v1alpha for Gemini Developer API Client (Python)
DESCRIPTION: This snippet configures the `genai.Client` for the Gemini Developer API to use the `v1alpha` API endpoints. This is done by providing `http_options` with `api_version='v1alpha'` during client initialization, alongside the required `api_key`.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_8

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

# Only run this block for Gemini Developer API
client = genai.Client(
    api_key='GEMINI_API_KEY',
    http_options=types.HttpOptions(api_version='v1alpha')
)
```

----------------------------------------

TITLE: Setting API Version to v1alpha for Gemini Developer API Client
DESCRIPTION: This snippet shows how to set the API version to `v1alpha` for the Gemini Developer API client. This allows access to specific preview features or older API versions as needed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_8

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client(
    api_key='GEMINI_API_KEY',
    http_options=types.HttpOptions(api_version='v1alpha')
)
```

----------------------------------------

TITLE: Creating a Batch Prediction Job in Google GenAI Python
DESCRIPTION: This snippet demonstrates how to create a new batch prediction job using `client.batches.create`. It specifies the model and source data, with destination and job display name being auto-populated. This feature is currently only supported in Vertex AI.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_78

LANGUAGE: python
CODE:
```
# Specify model and source file only, destination and job display name will be auto-populated
job = client.batches.create(
    model='gemini-2.0-flash-001',
    src='bq://my-project.my-dataset.my-table',
)

job
```

----------------------------------------

TITLE: Disabling Automatic Function Calling - Python
DESCRIPTION: This snippet shows how to explicitly disable automatic function calling when passing a Python function as a tool. By setting `automatic_function_calling.disable` to `True`, the model will return function call parts in the response instead of executing the function automatically.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_27

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
  model='gemini-2.0-flash-001',
  contents='What is the weather like in Boston?',
  config=types.GenerateContentConfig(
    tools=[get_current_weather],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
      disable=True
    ),
  ),
)
```

----------------------------------------

TITLE: Polling Tuning Job Status in Gemini API (Python)
DESCRIPTION: This snippet continuously polls the status of a tuning job until it is no longer in a running or pending state. It uses `time.sleep` to pause between checks and prints the current job state, useful for monitoring long-running operations.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_66

LANGUAGE: python
CODE:
```
import time

running_states = set(
    [
        'JOB_STATE_PENDING',
        'JOB_STATE_RUNNING',
    ]
)

while tuning_job.state in running_states:
    print(tuning_job.state)
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(10)
```

----------------------------------------

TITLE: Listing Batch Jobs with Pager (Asynchronous) - Python
DESCRIPTION: This asynchronous example demonstrates listing batch prediction jobs using an asynchronous pager object. It retrieves the first page, prints its size and the first item, then asynchronously advances to the next page and prints its first item.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_84

LANGUAGE: python
CODE:
```
from google.genai import types

async_pager = await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
)
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Listing Batch Jobs (Asynchronous Iteration) - Python
DESCRIPTION: This asynchronous snippet shows how to iterate through a paginated list of batch prediction jobs. It uses `client.aio.batches.list` with a `ListBatchJobsConfig` for page size. Each batch job is then printed asynchronously.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_83

LANGUAGE: python
CODE:
```
from google.genai import types

async for job in await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
):
    print(job)
```

----------------------------------------

TITLE: Retrieving Tuned Model Information from Gemini API (Python)
DESCRIPTION: This snippet retrieves detailed information about a specific tuned model using `client.models.get`. It requires the `model` identifier, which can be obtained from the `tuning_job.tuned_model.model` attribute, providing access to its metadata.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_68

LANGUAGE: python
CODE:
```
tuned_model = client.models.get(model=tuning_job.tuned_model.model)
print(tuned_model)
```

----------------------------------------

TITLE: Preparing Cache Content for Gemini API (Python)
DESCRIPTION: This snippet prepares the `file_uris` for creating cached content, conditionally selecting between GCS URIs for Vertex AI or local file URIs for Gemini Developer API. It imports `types` for defining content parts from URIs, ensuring compatibility across environments.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_59

LANGUAGE: python
CODE:
```
from google.genai import types

if client.vertexai:
    file_uris = [
        'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
        'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
    ]
else:
    file_uris = [file1.uri, file2.uri]
```

----------------------------------------

TITLE: Listing Batch Jobs (Synchronous Iteration) - Python
DESCRIPTION: This snippet shows how to iterate through a paginated list of batch prediction jobs. It uses `client.batches.list` with a `ListBatchJobsConfig` to specify page size. Each batch job in the list is then printed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_81

LANGUAGE: python
CODE:
```
from google.genai import types

for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
    print(job)
```

----------------------------------------

TITLE: Listing Batch Prediction Jobs Synchronously in Google GenAI Python
DESCRIPTION: This example demonstrates how to list batch prediction jobs using `client.batches.list` with a `types.ListBatchJobsConfig` for pagination. It iterates and prints each job found, providing an overview of active and completed jobs.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_81

LANGUAGE: python
CODE:
```
for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
    print(job)
```

----------------------------------------

TITLE: Listing Tuning Jobs (Synchronous Iteration) - Python
DESCRIPTION: This snippet shows how to iterate through a paginated list of tuning jobs. It uses `client.tunings.list` with a configuration to specify page size. Each tuning job in the list is then printed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_74

LANGUAGE: python
CODE:
```
for job in client.tunings.list(config={'page_size': 10}):
    print(job)
```

----------------------------------------

TITLE: Retrieving File Information from Gemini API
DESCRIPTION: First uploads a file, then demonstrates how to retrieve its metadata and information from the Gemini API using the `client.files.get` method, identifying the file by its `name` attribute.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_57

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file_info = client.files.get(name=file1.name)
```

----------------------------------------

TITLE: Deleting a Batch Prediction Job in Google GenAI Python
DESCRIPTION: This snippet shows how to delete a specific batch prediction job resource using its name via `client.batches.delete`. The `delete_job` variable will hold the response from the deletion operation, confirming the action.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_85

LANGUAGE: python
CODE:
```
# Delete the job resource
delete_job = client.batches.delete(name=job.name)

delete_job
```

----------------------------------------

TITLE: Retrieving a Tuned Model (Synchronous) - Python
DESCRIPTION: This snippet demonstrates how to retrieve a specific tuned model using its identifier. It uses the `client.models.get` method, passing the model name obtained from a `tuning_job` object. The retrieved model object is then printed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_67

LANGUAGE: python
CODE:
```
tuned_model = client.models.get(model=tuning_job.tuned_model.model)
print(tuned_model)
```

----------------------------------------

TITLE: Editing Images with Imagen in Python
DESCRIPTION: This snippet demonstrates how to edit an image using the `edit_image` method, which utilizes a different model than generation or upscaling. It involves creating `RawReferenceImage` and `MaskReferenceImage` objects to define the editing context, specifying a prompt, and configuring the edit mode and output. This feature is exclusive to Vertex AI.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_49

LANGUAGE: python
CODE:
```
# Edit the generated image from above
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage

raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-001',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode='EDIT_MODE_INPAINT_INSERTION',
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response3.generated_images[0].image.show()
```

----------------------------------------

TITLE: Retrieving Batch Job State by Name (Synchronous) - Python
DESCRIPTION: This example shows how to retrieve a specific batch prediction job by its name. It uses `client.batches.get` and then accesses and prints the `state` attribute of the retrieved job object.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_79

LANGUAGE: python
CODE:
```
# Get a job by name
job = client.batches.get(name=job.name)

job.state
```

----------------------------------------

TITLE: Setting API Version to v1 for Vertex AI Client
DESCRIPTION: This snippet demonstrates how to explicitly set the API version to `v1` for a Vertex AI client using `http_options`. This is useful for selecting stable API endpoints instead of the default beta features.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_7

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1',
    http_options=types.HttpOptions(api_version='v1')
)
```

----------------------------------------

TITLE: Updating a Tuned Model (Synchronous) - Python
DESCRIPTION: This example shows how to update the configuration of an existing tuned model, specifically its display name and description. It utilizes `client.models.update` with an `UpdateModelConfig` object to specify the changes. The updated model object is then printed.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_68

LANGUAGE: python
CODE:
```
from google.genai import types

tuned_model = client.models.update(
    model=tuning_job.tuned_model.model,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    ),
)
print(tuned_model)
```

----------------------------------------

TITLE: Updating a Tuned Model in Google GenAI Python
DESCRIPTION: This snippet demonstrates how to update an existing tuned model's display name and description using the `client.models.update` method. It requires an existing model object (e.g., `pager[0]`) and uses `types.UpdateModelConfig` for specifying the updates.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_73

LANGUAGE: python
CODE:
```
from google.genai import types

model = pager[0]

model = client.models.update(
    model=model.name,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    ),
)

print(model)
```

----------------------------------------

TITLE: Retrieving Tuning Job Information
DESCRIPTION: Retrieves the current status and details of a specific model tuning job using its `name` attribute. This allows monitoring the progress or outcome of the tuning operation.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_64

LANGUAGE: python
CODE:
```
tuning_job = client.tunings.get(name=tuning_job.name)
    print(tuning_job)
```

----------------------------------------

TITLE: Retrieving File Information from Gemini API (Python)
DESCRIPTION: This snippet first uploads a PDF file and then retrieves its metadata using `client.files.get` by providing the `name` attribute of the uploaded file. This demonstrates how to access information about previously uploaded files, such as their URI or status.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_57

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file_info = client.files.get(name=file1.name)
```

----------------------------------------

TITLE: Retrieving Tuning Job Information from Gemini API (Python)
DESCRIPTION: This snippet demonstrates how to retrieve the current status and details of a tuning job using `client.tunings.get`. It requires the `name` attribute of the tuning job to fetch its progress and configuration.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_65

LANGUAGE: python
CODE:
```
tuning_job = client.tunings.get(name=tuning_job.name)
print(tuning_job)
```

----------------------------------------

TITLE: Retrieving Cached Content Information
DESCRIPTION: Retrieves the details of a previously created cached content entry using its `name` attribute. This allows for inspection or verification of the cached content's status and configuration.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_60

LANGUAGE: python
CODE:
```
cached_content = client.caches.get(name=cached_content.name)
```

----------------------------------------

TITLE: Deleting Files from Gemini API
DESCRIPTION: Uploads a file to the Gemini API, then demonstrates how to delete it using the `client.files.delete` method. The file is referenced by its `name` attribute.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_58

LANGUAGE: python
CODE:
```
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)
```

----------------------------------------

TITLE: Deleting a Batch Prediction Job (Synchronous) - Python
DESCRIPTION: This snippet demonstrates how to delete a specific batch prediction job resource by its name. It uses `client.batches.delete` and then prints the result of the deletion operation.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_85

LANGUAGE: python
CODE:
```
# Delete the job resource
delete_job = client.batches.delete(name=job.name)

delete_job
```

----------------------------------------

TITLE: Upscaling Images with Imagen Model in Python (Vertex AI)
DESCRIPTION: This example shows how to upscale a previously generated image using the `upscale_image` method. It specifies the upscale factor and output configuration, noting that this feature is currently only supported in Vertex AI.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_48

LANGUAGE: Python
CODE:
```
from google.genai import types

# Upscale the generated image from above
response2 = client.models.upscale_image(
    model='imagen-3.0-generate-001',
    image=response1.generated_images[0].image,
    upscale_factor='x2',
    config=types.UpscaleImageConfig(
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response2.generated_images[0].image.show()
```

----------------------------------------

TITLE: Upscaling Images with Imagen in Python
DESCRIPTION: This example shows how to upscale a previously generated image using the `upscale_image` method. It takes an existing image object, an `upscale_factor`, and configuration options like `include_rai_reason` and `output_mime_type`. This functionality is currently supported only in Vertex AI.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_48

LANGUAGE: python
CODE:
```
from google.genai import types

# Upscale the generated image from above
response2 = client.models.upscale_image(
    model='imagen-3.0-generate-002',
    image=response1.generated_images[0].image,
    upscale_factor='x2',
    config=types.UpscaleImageConfig(
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response2.generated_images[0].image.show()
```

----------------------------------------

TITLE: Structuring Content with String Input
DESCRIPTION: This snippet illustrates how a simple string input for the `contents` argument is automatically converted by the SDK. It assumes the string is a text part and wraps it in a `types.UserContent` instance.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_13

LANGUAGE: python
CODE:
```
contents='Why is the sky blue?'
```

----------------------------------------

TITLE: Copying PDF Files with gsutil (Command Line)
DESCRIPTION: This command-line snippet uses `gsutil cp` to copy two PDF files from a Google Cloud Storage (GCS) bucket to the local directory. These files are prerequisites for subsequent file operations demonstrated with the Gemini Developer API.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_55

LANGUAGE: cmd
CODE:
```
!gsutil cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
!gsutil cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .
```

----------------------------------------

TITLE: Downloading Sample PDF Files (Console)
DESCRIPTION: Uses the `gsutil` command-line tool to copy two sample PDF files from a Google Cloud Storage bucket to the current local directory. These files are prerequisites for subsequent file upload and caching examples.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_55

LANGUAGE: console
CODE:
```
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .
```

----------------------------------------

TITLE: Downloading File for Content Generation Example
DESCRIPTION: This snippet shows how to download a sample text file (`a11.txt`) using `wget`. This file is then used in a subsequent example to demonstrate generating content from an uploaded file.
SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_10

LANGUAGE: sh
CODE:
```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

----------------------------------------

TITLE: Downloading a File for Upload (Console)
DESCRIPTION: This console command uses `wget` to download a sample text file (`a11.txt`) from a Google Cloud Storage bucket. This file is intended to be used as an input for content generation, particularly with the Gemini Developer API's file upload feature.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_10

LANGUAGE: console
CODE:
```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

----------------------------------------

TITLE: API Reference for genai.chats Module
DESCRIPTION: This entry describes the API documentation for the `genai.chats` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for chat-related functionalities.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_3

LANGUAGE: APIDOC
CODE:
```
Module: genai.chats
  Purpose: Provides functionalities for chat interactions within the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.tokens Module
DESCRIPTION: This entry describes the API documentation for the `genai.tokens` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for token-related functionalities.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_7

LANGUAGE: APIDOC
CODE:
```
Module: genai.tokens
  Purpose: Provides functionalities related to token handling and management in the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.types Module
DESCRIPTION: This entry describes the API documentation for the `genai.types` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for data types and structures.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_9

LANGUAGE: APIDOC
CODE:
```
Module: genai.types
  Purpose: Defines common data types and structures used across the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.live Module
DESCRIPTION: This entry describes the API documentation for the `genai.live` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for live interaction functionalities.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_5

LANGUAGE: APIDOC
CODE:
```
Module: genai.live
  Purpose: Functionalities for live interactions and real-time operations in the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.models Module
DESCRIPTION: This entry describes the API documentation for the `genai.models` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for model-related operations.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_6

LANGUAGE: APIDOC
CODE:
```
Module: genai.models
  Purpose: API for interacting with and managing models in the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.batches Module
DESCRIPTION: This entry describes the API documentation for the `genai.batches` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for batch processing functionalities.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_1

LANGUAGE: APIDOC
CODE:
```
Module: genai.batches
  Purpose: Functionalities related to batch processing within the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.files Module
DESCRIPTION: This entry describes the API documentation for the `genai.files` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for file management operations.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_4

LANGUAGE: APIDOC
CODE:
```
Module: genai.files
  Purpose: API for managing files and file-related operations in the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.client Module
DESCRIPTION: This entry describes the API documentation for the `genai.client` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for client-side functionalities.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_0

LANGUAGE: APIDOC
CODE:
```
Module: genai.client
  Purpose: Client-side operations for the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.caches Module
DESCRIPTION: This entry describes the API documentation for the `genai.caches` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for caching mechanisms.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_2

LANGUAGE: APIDOC
CODE:
```
Module: genai.caches
  Purpose: API for managing caching mechanisms in the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```

----------------------------------------

TITLE: API Reference for genai.tunings Module
DESCRIPTION: This entry describes the API documentation for the `genai.tunings` module. It covers all public and undocumented members, along with their inheritance relationships, providing a comprehensive reference for model tuning functionalities.
SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/genai.rst.txt#_snippet_8

LANGUAGE: APIDOC
CODE:
```
Module: genai.tunings
  Purpose: API for managing and applying tunings to models in the GenAI library.
  API Documentation Scope:
    - All public members
    - Undocumented members
    - Inheritance hierarchy
```