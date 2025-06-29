"""
WatsonX Generator Example

This example demonstrates how to use the WatsonxGenerator component with the Haystack framework.
It shows basic usage, advanced configuration, streaming, pipeline integration, and async operations.

Prerequisites:
- Set environment variables: WATSONX_API_KEY and WATSONX_PROJECT_ID
- Install: pip install haystack-watsonx
"""

import asyncio
import os

from haystack import Pipeline
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret

from haystack_integrations.components.generators.watsonx.generator import WatsonxGenerator


def basic_usage_example():
    """Demonstrates basic usage of WatsonxGenerator"""
    print("=== Basic Usage Example ===")

    generator = WatsonxGenerator(
        model="meta-llama/llama-3-3-70b-instruct",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
        generation_kwargs={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9, "decoding_method": "sample"},
    )

    system_prompt = "You are a helpful assistant. Provide clear, direct answers."

    questions = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "List benefits of renewable energy.",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = generator.run(prompt=question, system_prompt=system_prompt)

        response = result["replies"][0]
        metadata = result["meta"][0]

        print(f"Answer: {response}")
        print(f"Tokens used: {metadata.get('usage', {}).get('total_tokens', 'N/A')}")
        print("-" * 60)


def advanced_configuration_example():
    """Shows advanced configuration options"""
    print("\n=== Advanced Configuration Example ===")

    generator = WatsonxGenerator(
        model="meta-llama/llama-3-3-70b-instruct",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
        generation_kwargs={
            "max_tokens": 512,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "decoding_method": "sample",
            "stop_sequences": ["Human:", "Assistant:"],
        },
    )

    system_prompt = "You are a creative writing assistant. Write engaging, vivid descriptions."
    prompt = "Describe a peaceful mountain lake at sunrise."

    result = generator.run(
        prompt=prompt, system_prompt=system_prompt, generation_kwargs={"temperature": 0.9, "max_tokens": 512}
    )

    print(f"Creative writing prompt: {prompt}")
    print(f"Response: {result['replies'][0]}")
    print(f"Metadata: {result['meta'][0]}")


def streaming_example():
    """Demonstrates streaming responses"""
    print("\n=== Streaming Example ===")

    generator = WatsonxGenerator(
        model="meta-llama/llama-3-3-70b-instruct",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
        generation_kwargs={"max_tokens": 512, "temperature": 0.7},
    )

    def streaming_callback(chunk: StreamingChunk):
        """Handle streaming chunks as they arrive"""
        print(chunk.content, end="", flush=True)

    system_prompt = "You are a knowledgeable science teacher. Explain concepts clearly."
    prompt = "Explain how rainbows form."

    print(f"Question: {prompt}")
    print("Streaming response: ", end="")

    result = generator.run(prompt=prompt, system_prompt=system_prompt, streaming_callback=streaming_callback)

    print("\n\nComplete response received.")
    print(f"Total tokens: {result['meta'][0].get('usage', {}).get('total_tokens', 'N/A')}")


def pipeline_integration_example():
    """Shows how to use WatsonxGenerator in a Haystack pipeline"""
    print("\n=== Pipeline Integration Example ===")

    generator = WatsonxGenerator(
        model="meta-llama/llama-3-3-70b-instruct",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
        generation_kwargs={"max_tokens": 512, "temperature": 0.6},
    )

    # Create and configure pipeline
    pipeline = Pipeline()
    pipeline.add_component("generator", generator)

    system_prompt = "You are an expert historian. Provide accurate, concise historical information."
    prompt = "Who built the Great Wall of China and why?"

    result = pipeline.run(
        {"generator": {"prompt": prompt, "system_prompt": system_prompt, "generation_kwargs": {"temperature": 0.5}}}
    )

    print(f"Pipeline question: {prompt}")
    print(f"Pipeline response: {result['generator']['replies'][0]}")
    print(f"Pipeline metadata: {result['generator']['meta'][0]}")


async def async_example():
    """Demonstrates async usage of WatsonxGenerator"""
    print("\n=== Async Example ===")

    generator = WatsonxGenerator(
        model="meta-llama/llama-3-3-70b-instruct",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
        generation_kwargs={"max_tokens": 512, "temperature": 0.7},
    )

    system_prompt = "You are a helpful assistant providing practical advice."
    questions = [
        "What are three tips for better sleep?",
        "How can I reduce stress at work?",
        "What's a healthy breakfast idea?",
    ]

    tasks = []
    for question in questions:
        task = generator.run_async(prompt=question, system_prompt=system_prompt)
        tasks.append((question, task))

    print("Processing multiple questions asynchronously...")

    for question, task in tasks:
        result = await task
        print(f"\nQ: {question}")
        print(f"A: {result['replies'][0]}")


def main():
    """Run all examples"""
    print("WatsonX Generator Examples for Haystack")
    print("=" * 50)

    if not os.getenv("WATSONX_API_KEY") or not os.getenv("WATSONX_PROJECT_ID"):
        print("Please set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables")
        return

    try:
        basic_usage_example()
        advanced_configuration_example()
        streaming_example()
        pipeline_integration_example()

        print("\nRunning async example...")
        asyncio.run(async_example())

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\n Example failed: {e}")
        print("Make sure your WatsonX credentials are correctly set.")


if __name__ == "__main__":
    main()

# WatsonX Generator Examples for Haystack
# ==================================================
# === Basic Usage Example ===

# Question: What is the capital of France?
# Answer: The capital of France is Paris.
# Tokens used: 62
# ------------------------------------------------------------

# Question: Explain photosynthesis in one sentence.
# Answer: Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun
# into chemical energy in the form of glucose, releasing oxygen as a byproduct.
# Tokens used: 92
# ------------------------------------------------------------

# Question: List benefits of renewable energy.
# Answer: The benefits of renewable energy include:

# 1. **Sustainability**: Renewable energy sources are replenished naturally and are not depleted over time.
# 2. **Reduced Greenhouse Gas Emissions**: Renewable energy sources produce little to no greenhouse gas emissions,
# contributing to a decrease in climate change.
# 3. **Lower Operating Costs**: Renewable energy sources have lower operating costs compared to traditional fossil
# fuel-based power plants.
# 4. **Energy Independence**: Renewable energy can reduce reliance on imported fuels, improving energy security and
# reducing dependence on foreign energy.

# Tokens used: 153
# ------------------------------------------------------------

# === Advanced Configuration Example ===
# Creative writing prompt: Describe a peaceful mountain lake at sunrise.

# Response: As the first whispers of dawn caressed the sky, the mountain lake awoke from its slumber, its tranquil
# surface mirroring the heavens like a sheet of polished glass. The surrounding peaks, still cloaked in a soft veil of
# night, stood sentinel, their rugged silhouettes etched against a canvas of pink and gold hues that deepened with each
# passing moment.

# As sunrise approached, the sky transformed into a kaleidoscope of warm colors - coral, salmon, and amber - which
# danced across the water, casting a mesmerizing glow on the lake's surface. The gentle lapping of the water against
# the shore created a soothing melody, a lullaby that seemed to lull the world into a state of serenity.

# A delicate mist, born from the lake's nocturnal slumber, began to rise, weaving an ethereal veil that swirled and
# eddied above the water. The air was filled with the sweet scent of pine and the earthy aroma of damp soil, as if the
# forest itself was exhaling a sigh of contentment.

# As the sun slowly ascended, its rays kissed the lake's surface, creating a dazzling display of light and color. The
# water's glassy calm was broken only by the occasional ripple, as if a fish had stirred beneath the surface, leaving
# behind a trail of concentric circles that gradually dissipated into the stillness.

# The surrounding landscape, bathed in the warm, golden light of dawn, was set ablaze with color - the emerald green of
# the trees, the rusty red of the rocks, and the soft, velvety texture of the wildflowers that dotted the shore. The
# atmosphere was alive with the sweet songs of birds, their melodies intertwining with the gentle lapping of the water
# to create a symphony of peace and tranquility.

# In this fleeting moment, time stood still, and the world seemed to hold its breath, basking in the serenity of the
# mountain lake at sunrise. The beauty of this scene was almost palpable, a living, breathing entity that wrapped
# itself around the heart, filling the soul with a sense of wonder, awe, and deep, abiding peace.

# Metadata: {'model': 'meta-llama/llama-3-3-70b-instruct', 'index': 0, 'finish_reason': 'stop',
# 'usage': {'completion_tokens': 441, 'prompt_tokens': 56, 'total_tokens': 497}}

# === Streaming Example ===
# Question: Explain how rainbows form.

# Streaming response: Rainbows are one of the most beautiful and fascinating natural wonders. They occur when sunlight
# passes through water droplets in the air, and the process involves a combination of refraction, dispersion, and
# reflection. Let me break it down step by step:

# **Step 1: Sunlight enters the water droplet**
# When sunlight enters a water droplet, such as a cloud or mist, it is refracted, or bent, as it passes from air into
# the water. This is because light travels at different speeds in air and water.

# **Step 2: Dispersion occurs**
# As the sunlight is refracted, it is also split into its individual colors, a process known as dispersion. This is
# because each color of light has a slightly different wavelength and is refracted at a slightly different angle. This
# is why we see a band of colors in a rainbow, rather than just a single color.

# **Step 3: Reflection occurs**
# The dispersed light is then reflected off the back of the water droplet and bounces back towards the front of the
# droplet. This is known as total internal reflection.

# **Step 4: Refraction occurs again**
# As the reflected light exits the water droplet, it is refracted again, or bent, as it passes from water back into
# air. This second refraction causes the light to spread out and form a band of colors, which we see as a rainbow.

# **The position of the observer is crucial**
# For a rainbow to be visible, the observer must be in a specific position relative to the sun and the water droplets.
# The sun must be behind the observer, and the water droplets must be in front of them. The angle between the sun, the
# observer, and the water droplets is also critical, typically around 42 degrees.

# **The colors of the rainbow**
# The colors of the rainbow, often remembered using the acronym ROYGBIV, appear in the following order: Red, Orange,
# Yellow, Green, Blue, Indigo, and Violet. This is because the different wavelengths of light are refracted at slightly
# different angles, with red light being refracted at the smallest angle and violet light being refracted at the largest
# angle.

# In summary, rainbows form when sunlight passes through water droplets in the air, is refracted, dispersed, reflected,
# and refracted again, creating a beautiful band of colors that we see in the sky. The position of the observer and the
# angle of the sun are critical in determining the visibility and position of the rainbow.

# Complete response received.
# Total tokens: N/A

# === Pipeline Integration Example ===
# Pipeline question: Who built the Great Wall of China and why?

# Pipeline response: The Great Wall of China was built in multiple stages by several Chinese dynasties. The initial
# versions of the wall were constructed as early as the 7th century BC, during the Chu State period. However, the most
# famous and well-preserved versions were built during the following dynasties:

# 1. **Qin Dynasty (221-206 BC)**: Qin Shi Huang, the first emperor of China, ordered the construction of a long wall
# to protect his empire from invasions by nomadic tribes. This wall was approximately 3,000 miles (4,800 km) long.

# 2. **Han Dynasty (206 BC-220 AD)**: The Han Dynasty extended and fortified the wall, making it around 6,000 miles
# (9,656 km) long.

# 3. **Sui and Ming Dynasties (581-1644 AD)**: The Sui and Ming Dynasties renovated and extended the wall, with the Ming
# Dynasty building the most well-known and well-preserved version, which is around 4,000 miles (6,400 km) long.

# The primary purpose of the Great Wall of China was to:

# * **Protect the Chinese Empire from invasions**: The wall was built to prevent nomadic tribes, such as the Mongols
# and the Xiongnu, from invading and raiding Chinese territories.

# * **Control trade and immigration**: The wall helped to regulate trade and immigration, allowing the Chinese
# government to monitor and tax goods and people entering the country.

# * **Demonstrate the power and prestige of the Chinese Empire**: The wall was also a symbol of the Chinese Empire's
# power, wealth, and engineering prowess.

# Overall, the Great Wall of China is an impressive architectural achievement that showcases the ingenuity and labor
# of thousands of workers over several centuries.

# Pipeline metadata: {'model': 'meta-llama/llama-3-3-70b-instruct', 'index': 0, 'finish_reason': 'stop',
# 'usage': {'completion_tokens': 362, 'prompt_tokens': 58, 'total_tokens': 420}}

# Running async example...

# === Async Example ===
# Processing multiple questions asynchronously...

# Q: What are three tips for better sleep?
# A: Here are three tips for better sleep:

# 1. **Establish a Consistent Sleep Schedule**: Go to bed and wake up at the same time every day, including weekends.
# This helps regulate your body's internal clock and can improve the quality of your sleep. Aim for 7-9 hours of sleep
# each night.

# 2. **Create a Sleep-Conducive Environment**: Make your bedroom a sleep haven by ensuring it is dark, quiet, and cool.
# Consider using blackout curtains, earplugs, or a white noise machine if necessary. Invest in a comfortable mattress
# and pillows, and keep electronic devices out of the bedroom.

# 3. **Develop a Relaxing Bedtime Routine**: Wind down before bed with a calming activity, such as reading a book,
# taking a warm bath, or practicing gentle stretches or meditation. Avoid screens (e.g., phones, tablets, or laptops)
# for at least an hour before bedtime, as the blue light they emit can interfere with your sleep. A consistent bedtime
# routine can signal to your brain that it's time to sleep, making it easier to fall asleep and stay asleep.

# Remember, it may take some time to notice improvements in your sleep quality, so be patient and try to make these
# tips a consistent part of your daily routine.

# Q: How can I reduce stress at work?
# A: Reducing stress at work is essential for maintaining your overall well-being and productivity. Here are some
# practical tips to help you manage stress at work:

# 1. **Prioritize tasks**: Make a to-do list and focus on the most important tasks first. Break down large tasks into
# smaller, manageable chunks to help you stay organized and in control.

# 2. **Take breaks**: Take short breaks throughout the day to stretch, move around, and rest your mind. Use this time
# to do something you enjoy, such as taking a walk or practicing deep breathing exercises.

# 3. **Communicate with your team**: Don't be afraid to ask for help or support from your colleagues. Communicate your
# workload and deadlines with your team to ensure everyone is on the same page.

# 4. **Set boundaries**: Learn to say "no" to tasks that are not essential or that you simply cannot fit into your
# schedule. Set realistic expectations with your manager and colleagues to avoid overcommitting.

# 5. **Stay organized**: Keep your workspace organized and clutter-free. This will help you stay focused and avoid
# wasting time searching for lost documents or supplies.

# 6. **Practice self-care**: Take care of your physical and emotional needs by getting enough sleep, exercising
# regularly, and eating a healthy diet.

# 7. **Limit distractions**: Minimize distractions by turning off notifications, finding a quiet workspace, or using
# noise-cancelling headphones.

# 8. **Seek support**: If you're feeling overwhelmed, don't hesitate to seek support from a supervisor, HR
# representative, or a mental health professional.

# 9. **Set realistic goals**: Set achievable goals and celebrate your successes. This will help you stay motivated and
# focused.

# 10. **Disconnect from work**: Establish a clear boundary between your work and personal life by avoiding work-related
# tasks outside of work hours.

# 11. **Use stress-reducing techniques**: Try techniques such as meditation, yoga, or deep breathing exercises to help
# manage stress and anxiety.

# 12. **Take time off**: Use your vacation days or take a mental health day if you need to recharge and reduce stress.

# Remember, everyone experiences stress at work, but by implementing these tips, you can reduce your stress levels and
# improve your overall well-being.

# Which of these tips resonates with you the most, or is there something specific that's causing you stress at work? I'm
# here to help you brainstorm and find solutions.

# Q: What's a healthy breakfast idea?
# A: Here's a simple and nutritious breakfast idea:

# **Avocado Toast with Scrambled Eggs and Cherry Tomatoes**

# Ingredients:

# * 1 slice whole grain bread (e.g., whole wheat or multigrain)
# * 1/2 avocado, mashed
# * 2 scrambled eggs
# * 1/2 cup cherry tomatoes, halved
# * Salt and pepper to taste
# * Optional: 1 tablespoon olive oil, 1 tablespoon chopped fresh herbs (e.g., parsley, basil)

# Instructions:

# 1. Toast the bread until lightly browned.
# 2. Spread the mashed avocado on top of the toast.
# 3. Scramble the eggs in a bowl and cook them in a non-stick pan until set.
# 4. Place the scrambled eggs on top of the avocado.
# 5. Add the halved cherry tomatoes on top of the eggs.
# 6. Season with salt, pepper, and a drizzle of olive oil (if using).
# 7. Garnish with chopped fresh herbs (if using).

# This breakfast idea is packed with:

# * Whole grains (fiber and nutrients)
# * Healthy fats (avocado and olive oil)
# * Protein (eggs)
# * Fresh fruits (cherry tomatoes)
# * Fresh herbs (antioxidants and flavor)

# Feel free to customize this recipe to your taste preferences and dietary needs. Enjoy your delicious and nutritious
# breakfast!

# ==================================================
# All examples completed successfully!
