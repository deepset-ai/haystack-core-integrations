# In order to run this example, you will need watsonx credentials.
import asyncio
import os

from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.watsonx.chat.chat_generator import WatsonxChatGenerator

# Initialize the generator
generator = WatsonxChatGenerator(
    api_key=Secret.from_env_var('WATSONX_API_KEY'),  # Or use from_token("<apikey>")
    model='meta-llama/llama-3-2-1b-instruct',
    project_id=os.getenv('WATSONX_PROJECT_ID'),
    generation_kwargs={
        'max_tokens': 500,
        'temperature': 0.7,
        'top_p': 0.9,
    },
    tools=[
        {
            'name': 'get_weather',
            'description': 'Get the current weather in a given location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'},
                    'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit'], 'default': 'celsius'},
                },
                'required': ['location'],
            },
        }
    ],
    max_retries=3,
    timeout=30.0,
)


# Example 1: Basic synchronous chat
def basic_chat():
    messages = [
        ChatMessage.from_system('You are a helpful assistant.'),
        ChatMessage.from_user('Explain quantum computing in simple terms'),
    ]
    generator.run(messages)


# Example 2: Tool calling
def tool_calling_chat():
    messages = [ChatMessage.from_user("What's the weather in Berlin today?")]
    response = generator.run(messages)
    reply = response['replies'][0]
    if reply.tool_calls:
        for _tool_call in reply.tool_calls:
            pass
    else:
        pass


# Example 3: Streaming chat (sync)
def streaming_chat():
    messages = [ChatMessage.from_user('Write a short poem about artificial intelligence')]
    generator.run(messages, stream=True)


# Example 4: Asynchronous chat
def run_async_chat():
    async def _async_chat():
        messages = [ChatMessage.from_user('Tell me about the history of the internet')]
        await generator.run_async(messages)

    _run_async(_async_chat)


# Example 5: Asynchronous streaming chat
def run_async_streaming_chat():
    async def _async_streaming_chat():
        messages = [ChatMessage.from_user('Explain blockchain technology')]
        await generator.run_async(messages, stream=True)

    _run_async(_async_streaming_chat)


# Handle both notebook and script environments
def _run_async(coro_func):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.create_task(coro_func())
        else:
            return loop.run_until_complete(coro_func())
    except RuntimeError:
        import nest_asyncio

        nest_asyncio.apply()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro_func())


# Main
if __name__ == '__main__':
    basic_chat()
    tool_calling_chat()
    streaming_chat()
    run_async_chat()
    run_async_streaming_chat()


## Result
# === Watsonx Chat Generator Demo ===

# --- Basic Chat ---
# Quantum computing is a fascinating technology that uses the principles of quantum mechanics to perform calculations and operations on data. Here's a simple explanation:

# **Classical computers vs. Quantum Computers**

# A classical computer uses "bits" to store and process information. A bit is either 0 or 1, and it's like a light switch - it's either on or off. A quantum computer uses "qubits" (quantum bits), which can exist in multiple states at the same time. This means a quantum computer can process a vast number of possibilities simultaneously.

# **How Quantum Computers Work**

# Imagine you have a deck of cards with 52 cards in it. A classical computer would look at each card one by one, and if it finds a match, it would move on to the next card. A quantum computer, on the other hand, can look at all the cards simultaneously and find matches instantly.

# This is because qubits can exist in multiple states (0, 1, and both at the same time) and can be connected in a way that allows them to communicate with each other instantly, no matter how far apart they are.

# **Quantum Computing Steps**

# Here's a simplified step-by-step explanation of how a quantum computer works:

# 1. **Qubit Preparation**: The qubits are created and prepared in a special way to represent the problem we want to solve.
# 2. **Quantum Gates**: The qubits are then passed through a series of quantum gates, which are like the "switches" that control the qubits' behavior.
# 3. **Quantum Measurement**: The qubits are measured, and the outcome is determined.
# 4. **Data Processing**: The data is processed using the qubits' quantum gates, which allows for fast and efficient calculations.

# **What Quantum Computers Can Do**

# Quantum computers have the potential to solve certain problems much faster than classical computers. They can:

# * Break complex encryption codes
# * Simulate the behavior of molecules and materials
# * Optimize complex systems and processes
# * Discover new materials and compounds

# **Challenges and Limitations**

# While quantum computers have tremendous potential, they face several challenges and limitations, including:

# * **Error Correction**: Quantum computers are prone to errors due to the fragile nature of qubits.
# * **Scalability**: Currently, quantum computers are not scalable enough to solve complex problems.
# * **Quantum Noise**: Random fluctuations in qubits can affect the accuracy of quantum computations.

# Metadata: {'model': 'meta-llama/llama-3-2-1b-instruct', 'index': 0, 'finish_reason': 'length', 'usage': {'completion_tokens': 500, 'prompt_tokens': 48, 'total_tokens': 548}}

# --- Tool Calling Chat ---
# I'm not aware of any real-time weather information. However, I can suggest some ways for you to find out the current weather in Berlin. You can:

# * Check online weather websites such as AccuWeather, Weather.com, or the German national weather service, Deutscher Wetterdienst (DWD).
# * Use a mobile app such as Dark Sky or Weather Underground.
# * Tune into local news or radio stations for weather updates.

# If you'd like, I can help you find the most up-to-date weather information for Berlin.

# --- Streaming Chat ---

# Full message: I don't have the capability to write poems. However, I can assist you with generating a poem about artificial intelligence if you'd like.

# --- Async Chat ---
# Here's an overview of the history of the internet:

# *   **Early beginnings:** The internet has its roots in the 1960s, when the United States Department of Defense's Advanced Research Projects Agency (ARPA) funded a project to create a network of computers that could communicate with each other. This project, called ARPANET, was the first operational packet switching network.
# *   **Network expansion:** In the 1970s and 1980s, other networks, such as the National Science Foundation Network (NSFNET), were established, connecting universities and research institutions across the United States.
# *   **The World Wide Web (WWW):** In 1989, Tim Berners-Lee invented the World Wide Web, which revolutionized the way people access and share information online. The web made it easy for users to navigate and access information using web browsers and hyperlinks.
# *   **Internet growth:** The 1990s saw the rise of commercial internet services, such as America Online (AOL) and CompuServe. This led to the growth of the internet as a mainstream technology.
# *   **Globalization:** The internet became increasingly global, with the establishment of international networks and the development of new technologies, such as broadband and mobile internet.
# *   **Social media and online communities:** The 2000s saw the rise of social media platforms, such as Facebook and Twitter, which enabled people to connect with each other and share information online.
# *   **The internet today:** The internet is now a global network of billions of interconnected devices, with applications ranging from online shopping and education to social media and entertainment.

# Here's a summary of the key events and milestones in the history of the internet:

# *   1969: ARPANET is launched
# *   1989: The World Wide Web is invented
# *   1991: The internet is opened to commercial use
# *   1993: The World Wide Web is made available to the general public
# *   1995: Broadband internet becomes available
# *   2000: Social media platforms begin to emerge
# *   2007: Facebook becomes one of the most popular social media platforms
# *   2010s: The internet continues to evolve with the development of new technologies, such as mobile internet and artificial intelligence.

# --- Async Streaming Chat ---

# Full message: **Tool Call: Blockchain Technology**

# The tool call "Blockchain Technology" is a fundamental concept in the field of cryptography and distributed ledger technology. Here's a brief explanation:

# Blockchain technology is a decentralized, digital ledger that records transactions across a network of computers in a secure and transparent manner. It allows for the creation of a permanent, immutable record of all transactions, making it ideal for applications such as cryptocurrency, supply chain management, and voting systems.

# Here's a high-level overview of how blockchain technology works:

# 1.  **Distributed Ledger**: A blockchain is a distributed ledger that is maintained by a network of computers (nodes) that are connected to each other through a peer-to-peer network.
# 2.  **Transactions**: When a user wants to make a transaction, they create a new entry in the blockchain by broadcasting the transaction to the network of nodes.
# 3.  **Verification**: Each node verifies the transaction by checking its validity and ensuring that it is not tampered with.
# 4.  **Block Creation**: Once a node verifies the transaction, it creates a new block and adds it to the blockchain.
# 5.  **Blockchain Update**: Each node updates its copy of the blockchain with the new block, ensuring that all nodes have the same version of the blockchain.

# **Benefits of Blockchain Technology**

# 1.  **Security**: Blockchain technology is secure due to the use of cryptography and the decentralized nature of the network.
# 2.  **Transparency**: All transactions are recorded in a public ledger, making it transparent and tamper-proof.
# 3.  **Immutable**: The blockchain is an immutable record, meaning that once a transaction is recorded, it cannot be altered or deleted.
# 4.  **Decentralization**: Blockchain technology is decentralized, meaning that there is no single point of control or authority.

# **Real-World Applications of Blockchain Technology**

# 1.  **Cryptocurrencies**: Blockchain technology is the foundation for cryptocurrencies such as Bitcoin and Ethereum.
# 2.  **Supply Chain Management**: Blockchain technology is being used to track the movement of goods and materials throughout the supply chain.
# 3.  **Voting Systems**: Blockchain technology is being used to create secure and transparent voting systems.
# 4.  **Identity Verification**: Blockchain technology is being used to create secure and decentralized identity verification systems.

# In summary, blockchain technology is a decentralized, digital ledger that records transactions in a secure and transparent manner. Its benefits include security, transparency, immutability, and decentralization.
