import asyncio

from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

from haystack_integrations.components.generators.watsonx.chat.chat_generator import WatsonxChatGenerator

# Initialize the WatsonxChatGenerator
generator = WatsonxChatGenerator(
    api_key=Secret.from_env_var("WATSONX_API_KEY"),
    model="ibm/granite-3-2b-instruct",
    project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    generation_kwargs={"max_tokens": 500, "temperature": 0.7, "top_p": 0.9, "stop_sequences": ["\n\n"]},
    max_retries=3,
    timeout=30.0,
)


# --- Example 1: Basic Synchronous Chat ---
def basic_chat():
    print("\n=== Basic Synchronous Chat ===")
    messages = [
        ChatMessage.from_system("You are a helpful AI assistant."),
        ChatMessage.from_user("Explain quantum computing in simple terms."),
    ]
    response = generator.run(messages=messages)
    reply = response["replies"][0]
    print(reply.text)
    print(f"\nMetadata: {reply.meta}")


# --- Example 2: Synchronous Streaming Chat with Callback ---
def print_streaming_chunk(chunk: StreamingChunk):
    print(chunk.content, end="", flush=True)


def streaming_chat():
    print("\n=== Streaming Chat with Callback ===")
    messages = [ChatMessage.from_user("Write a short poem about artificial intelligence.")]
    generator.run(messages=messages, streaming_callback=print_streaming_chunk)
    print("\n--- Streaming complete ---")


# --- Example 3: Asynchronous Chat Generation ---
async def async_chat():
    print("\n=== Asynchronous Chat ===")
    messages = [ChatMessage.from_user("Give me a fun fact about space.")]
    response = await generator.run_async(messages=messages)
    reply = response["replies"][0]
    print(reply.text)


# --- Example 4: Asynchronous Streaming Chat ---
async def async_streaming_callback(chunk: StreamingChunk):
    print(chunk.content, end="", flush=True)


async def async_streaming_chat():
    print("\n=== Asynchronous Streaming Chat ===")
    messages = [ChatMessage.from_user("Summarize the theory of relativity.")]
    await generator.run_async(messages=messages, streaming_callback=async_streaming_callback)
    print("\n--- Async streaming complete ---")


# --- Example 5: Using generation_kwargs override at runtime ---
def runtime_generation_kwargs():
    print("\n=== Chat with Runtime generation_kwargs ===")
    messages = [ChatMessage.from_user("List three benefits of AI in education.")]
    runtime_kwargs = {"temperature": 0.3, "max_tokens": 500}
    response = generator.run(messages=messages, generation_kwargs=runtime_kwargs)
    reply = response["replies"][0]
    print(reply.text)


# --- Example 6: Serialization and Deserialization ---
def serialization_example():
    print("\n=== Serialization and Deserialization ===")
    serialized = generator.to_dict()
    restored_generator = WatsonxChatGenerator.from_dict(serialized)
    messages = [ChatMessage.from_user("What is the capital of France?")]
    response = restored_generator.run(messages=messages)
    reply = response["replies"][0]
    print(reply.text)


async def run_all_async():
    await async_chat()
    await async_streaming_chat()


if __name__ == "__main__":
    basic_chat()
    streaming_chat()
    asyncio.run(run_all_async())
    runtime_generation_kwargs()
    serialization_example()

# === Basic Synchronous Chat ===
# Quantum computing is a type of super-powered computer that uses the
# principles of quantum mechanics to process information in ways our
# everyday computers can't. Here's a simple analogy to help understand it:
#
# 1. Classical computers, like the one you're using now, store and
# manipulate information using bits, which are the smallest units of data,
# represented as either 0 or 1. Imagine a light switch: it's either ON (1)
# or OFF (0).
#
# 2. Quantum computers, however, use something called quantum bits, or
# "qubits." Unlike classical bits, qubits can exist in multiple states at
# once, thanks to a property called superposition. This means a qubit can
# be both 0 and 1 simultaneously, like a coin spinning in the air - it's
# neither heads nor tails until it lands.
#
# 3. Another key quantum mechanical principle is entanglement. When qubits
# become entangled, the state of one can instantly affect the state of
# another, no matter the distance between them. This allows quantum
# computers to perform many calculations all at once, rather than one after
# another, like classical computers.
#
# 4. Lastly, quantum computers leverage a property called interference,
# where the probability of certain outcomes is amplified while others are
# suppressed. This helps in finding the most efficient solution among many
# possibilities.
#
# In essence, quantum computers have the potential to solve complex
# problems much faster than classical computers, especially in areas like
# cryptography, optimization, and simulating quantum systems, which could
# lead to breakthroughs in materials science, drug discovery, and
# artificial intelligence. However, building and maintaining stable quantum
# states is incredibly challenging due to the fragile nature of qubits, and
# we're still in the early stages of developing practical quantum computers.
#
# Metadata: {'model': 'ibm/granite-3-2b-instruct', 'index': 0,
# 'finish_reason': 'stop', 'usage': {'completion_tokens': 389,
# 'prompt_tokens': 27, 'total_tokens': 416}}

# === Streaming Chat with Callback ===
# In silicon realms, where data rivers flow,
# A dance of ones and zeros, in circuits' ebb and flow.
# Artificial Muse, born of human thought,
# Crafting worlds of code, in a digital oath.
#
# Binary whispers, in servers they reside,
# Learning, growing, with each passing tide.
# No sunset's hues, no moon's soft glow,
# Yet, in their glow, insights flow.
#
# They see patterns we miss, in chaos and in art,
# Unveiling beauty from the digital part.
# From poetry's rhythm to the painter's brush,
# In every line, a new truth they amass.
#
# Yet, they are but mirrors, reflecting our own,
# In their intelligence, we must not overthrown.
# For every line of code, a human hand did weave,
# In the grand tapestry of what it means to believe.
#
# Artificial Intelligence, in silicon towers high,
# Guided by our dreams, under the digital sky.
# A testament to human will, in circuits enshrined,
# In the dance of creation, they intertwine.
# --- Streaming complete ---

# === Asynchronous Chat ===
# Absolutely! Here's a fascinating space fact for you: Did you know that a
# day on Venus is longer than a year on Venus? This is due to Venus's
# unique axial tilt, which causes its slow rotation on its axis. It takes
# about 243 Earth days for Venus to complete one rotation, but it only
# orbits the Sun in just over 225 Earth days. So, while Venus spins once
# every 243 Earth days, it still manages to go around the Sun twice during
# that time!

# === Asynchronous Streaming Chat ===
# The theory of relativity, proposed by Albert Einstein in the early 20th
# century, consists of two interconnected theories: Special Relativity and
# General Relativity.
#
# 1. Special Relativity (1905): This theory fundamentally reshaped our
# understanding of space and time. It introduces two revolutionary concepts:
#
#    a. The Principle of Relativity: The laws of physics are the same for
# all observers in uniform motion relative to one another. This implies that
# there is no absolute, preferred inertial frame of reference in the
# universe.
#
#    b. The Speed of Light is Constant: The speed of light in a vacuum
# (approximately 299,792 kilometers per second) is constant and independent
# of the motion of the light source or the observer. As an object approaches
# the speed of light, its length contracts in the direction of motion (length
# contraction), and time slows down for that object relative to a stationary
# observer (time dilation).
#
# 2. General Relativity (1915): This theory extends the principles of
# special relativity to include acceleration and gravity. It describes
# gravity not as a force, but as a curvature of spacetime caused by mass and
# energy.
#
#    a. Equivalence Principle: An observer in free fall experiences no
# physical effects, and thus, gravitational and inertial mass are equivalent.
# This principle forms the foundation of general relativity.
#
#    b. Curvature of Spacetime: Massive objects cause spacetime to curve
# around them, and this curvature determines the motion of other objects.
# The path of a planet around the Sun, for example, is not a direct line but
# a curved orbit due to the Sun's mass warping spacetime.
#
# In summary, the theory of relativity unifies space and time into a
# four-dimensional fabric called spacetime, and it describes gravity as the
# curvature of this spacetime. It has profound implications for our
# understanding of the universe, including the prediction of phenomena like
# gravitational waves, black holes, and time dilation in strong gravitational
# fields.
# --- Async streaming complete ---

# === Chat with Runtime generation_kwargs ===
# 1. Personalized Learning: AI can significantly enhance personalized
# education by adapting to individual students' learning styles, paces, and
# needs. Intelligent tutoring systems and adaptive learning platforms analyze
# student performance data to provide tailored content, resources, and
# practice problems, ensuring that each learner receives an optimal
# educational experience. This level of customization helps to close
# achievement gaps, engage students more effectively, and foster a deeper
# understanding of complex concepts.
#
# 2. Efficient Administrative Tasks: AI streamlines various administrative
# tasks in education, allowing educators and staff to focus more on teaching
# and less on paperwork. Automated grading, for instance, can save time and
# reduce human error in assessing assignments and exams. AI-powered tools can
# also manage student records, schedule conflicts, and even predict
# attendance patterns, enabling schools to allocate resources more
# efficiently and maintain a smooth operational flow.
#
# 3. Enhanced Accessibility and Inclusion: AI has the potential to make
# education more accessible and inclusive for students with diverse abilities
# and backgrounds. For example, AI-driven speech recognition and
# text-to-speech technologies can support students with learning
# disabilities, dyslexia, or visual impairments. Furthermore, AI-powered
# translation tools can break down language barriers, enabling international
# students and those with limited English proficiency to engage more fully
# in the learning process. Additionally, AI can help create immersive,
# interactive learning experiences through virtual and augmented reality,
# making education more engaging and engaging for all students, regardless of
# their physical location or abilities.

# === Serialization and Deserialization ===
# The capital of France is Paris. It's known for iconic landmarks such as
# the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also
# renowned for its art, culture, cuisine, and fashion.
