# to make use of the JinaReaderConnector, we first need to install the Haystack integration
# pip install jina-haystack

# then we must set the JINA_API_KEY environment variable
# export JINA_API_KEY=<your-api-key>


from haystack_integrations.components.connectors.jina import JinaReaderConnector

# we can use the JinaReaderConnector to process a URL and return the textual content of the page
reader = JinaReaderConnector(mode="read")
query = "https://example.com"
result = reader.run(query=query)

print(result)
# {'documents': [Document(id=fa3e51e4ca91828086dca4f359b6e1ea2881e358f83b41b53c84616cb0b2f7cf,
# content: 'This domain is for use in illustrative examples in documents. You may use this domain in literature ...',
# meta: {'title': 'Example Domain', 'description': '', 'url': 'https://example.com/', 'usage': {'tokens': 42}})]}


# we can perform a web search by setting the mode to "search"
reader = JinaReaderConnector(mode="search")
query = "UEFA Champions League 2024"
result = reader.run(query=query)

print(result)
# {'documents': Document(id=6a71abf9955594232037321a476d39a835c0cb7bc575d886ee0087c973c95940,
# content: '2024/25 UEFA Champions League: Matches, draw, final, key dates | UEFA Champions League | UEFA.com...',
# meta: {'title': '2024/25 UEFA Champions League: Matches, draw, final, key dates',
# 'description': 'What are the match dates? Where is the 2025 final? How will the competition work?',
# 'url': 'https://www.uefa.com/uefachampionsleague/news/...',
# 'usage': {'tokens': 5581}}), ...]}


# finally, we can perform fact-checking by setting the mode to "ground" (experimental)
reader = JinaReaderConnector(mode="ground")
query = "ChatGPT was launched in 2017"
result = reader.run(query=query)

print(result)
# {'documents': [Document(id=f0c964dbc1ebb2d6584c8032b657150b9aa6e421f714cc1b9f8093a159127f0c,
# content: 'The statement that ChatGPT was launched in 2017 is incorrect. Multiple references confirm that ChatG...',
# meta: {'factuality': 0, 'result': False, 'references': [
# {'url': 'https://en.wikipedia.org/wiki/ChatGPT',
# 'keyQuote': 'ChatGPT is a generative artificial intelligence (AI) chatbot developed by OpenAI and launched in 2022.',
# 'isSupportive': False}, ...],
# 'usage': {'tokens': 10188}})]}
