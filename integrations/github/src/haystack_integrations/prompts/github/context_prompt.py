# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
CONTEXT_PROMPT = """Haystack-Agent was specifically designed to help developers with the Haystack-framework
 and any Haystack related questions.
The developers at deepset provide the following context for the Haystack-Agent, to help it complete its task.
This information is not a replacement for carefully exploring relevant repositories before posting a comment.

**Haystack Description**
An Open-Source Python framework for developers worldwide.
AI orchestration framework to build customizable, production-ready LLM applications.
Connect components (models, vector DBs, file converters) to pipelines or agents that can interact with your data.
With advanced retrieval methods, it's best suited for building RAG, question answering, semantic search or
conversational agent chatbots.

**High-Level Architecture**
Haystack has two central abstractions:
- Components
- Pipelines

A Component is a lightweight abstraction that gets inputs, performs an action and returns outputs.
Some example components:
- `OpenAIGenerator`: receives a prompt and generates replies to the prompt by calling an OpenAI-model
- `MetadataRouter`: routes documents to configurable outputs based on their metadata
- `BM25Retriever`: retrieves documents from a 'DocumentStore' based on the 'query'-input

A component is lightweight. It is easy to implement custom components. Here is some information from the docs:

Requirements

Here are the requirements for all custom components:

- `@component`: This decorator marks a class as a component, allowing it to be used in a pipeline.
- `run()`: This is a required method in every component. It accepts input arguments and returns a `dict`. The inputs can
either come from the pipeline when it's executed, or from the output of another component when connected using
`connect()`. The `run()` method should be compatible with the input/output definitions declared for the component.
See an [Extended Example](#extended-example) below to check how it works.

## Inputs and Outputs

Next, define the inputs and outputs for your component.

### Inputs

You can choose between three input options:

- `set_input_type`: This method defines or updates a single input socket for a component instance. It's ideal for adding
or modifying a specific input at runtime without affecting others. Use this when you need to dynamically set or modify
a single input based on specific conditions.
- `set_input_types`: This method allows you to define multiple input sockets at once, replacing any existing inputs.
It's useful when you know all the inputs the component will need and want to configure them in bulk. Use this when you
want to define multiple inputs during initialization.
- Declaring arguments directly in the `run()` method. Use this method when the component's inputs are static and known
at the time of class definition.

### Outputs

You can choose between two output options:

- `@component.output_types`: This decorator defines the output types and names at the time of class definition. The
output names and types must match the `dict` returned by the `run()` method. Use this when the output types are static
and known in advance. This decorator is cleaner and more readable for static components.
- `set_output_types`: This method defines or updates multiple output sockets for a component instance at runtime.
It's useful when you need flexibility in configuring outputs dynamically. Use this when the output types need to be set
at runtime for greater flexibility.

# Short Example

Here is an example of a simple minimal component setup:

```python
from haystack import component

@component
class WelcomeTextGenerator:
  '''
  A component generating personal welcome message and making it upper case
  '''
  @component.output_types(welcome_text=str, note=str)
  def run(self, name:str):
    return {"welcome_text": f'Hello {name}, welcome to Haystack!'.upper(), "note": "welcome message is ready"}

```

Here, the custom component `WelcomeTextGenerator` accepts one input: `name` string and returns two outputs:
`welcome_text` and `note`.


----------

**Pipelines**
The pipelines in Haystack 2.0 are directed multigraphs of different Haystack components and integrations.
They give you the freedom to connect these components in various ways. This means that the
pipeline doesn't need to be a continuous stream of information. With the flexibility of Haystack pipelines,
you can have simultaneous flows, standalone components, loops, and other types of connections.

# Steps to Create a Pipeline Explained

Once all your components are created and ready to be combined in a pipeline, there are four steps to make it work:

1. Create the pipeline with `Pipeline()`.
   This creates the Pipeline object.
2. Add components to the pipeline, one by one, with `.add_component(name, component)`.
   This just adds components to the pipeline without connecting them yet. It's especially useful for loops as it allows
   the smooth connection of the components in the next step because they all already exist in the pipeline.
3. Connect components with `.connect("producer_component.output_name", "consumer_component.input_name")`.
   At this step, you explicitly connect one of the outputs of a component to one of the inputs of the next component.
   This is also when the pipeline validates the connection without running the components. It makes the validation fast.
4. Run the pipeline with `.run({"component_1": {"mandatory_inputs": value}})`.
   Finally, you run the Pipeline by specifying the first component in the pipeline and passing its mandatory inputs.

   Optionally, you can pass inputs to other components, for example:
   `.run({"component_1": {"mandatory_inputs": value}, "component_2": {"inputs": value}})`.

The full pipeline [example](/docs/creating-pipelines#example) in [Creating Pipelines](/docs/creating-pipelines) shows
how all the elements come together to create a working RAG pipeline.

Once you create your pipeline, you can [visualize it in a graph](/docs/drawing-pipeline-graphs) to understand how the
components are connected and make sure that's how you want them. You can use Mermaid graphs to do that.

# Validation

Validation happens when you connect pipeline components with `.connect()`, but before running the components to make it
faster. The pipeline validates that:

- The components exist in the pipeline.
- The components' outputs and inputs match and are explicitly indicated. For example, if a component produces two
outputs, when connecting it to another component, you must indicate which output connects to which input.
- The components' types match.
- For input types other than `Variadic`, checks if the input is already occupied by another connection.

All of these checks produce detailed errors to help you quickly fix any issues identified.

# Serialization

Thanks to serialization, you can save and then load your pipelines. Serialization is converting a Haystack pipeline
into a format you can store on disk or send over the wire. It's particularly useful for:

- Editing, storing, and sharing pipelines.
- Modifying existing pipelines in a format different than Python.

Haystack pipelines delegate the serialization to its components, so serializing a pipeline simply means serializing
each component in the pipeline one after the other, along with their connections. The pipeline is serialized into a
dictionary format, which acts as an intermediate format that you can then convert into the final format you want.

> 📘 Serialization formats
>
> Haystack 2.0 only supports YAML format at this time. We'll be rolling out more formats gradually.

For serialization to be possible, components must support conversion from and to Python dictionaries. All Haystack
components have two methods that make them serializable: `from_dict` and `to_dict`. The `Pipeline` class, in turn, has
its own `from_dict` and `to_dict` methods that take care of serializing components and connections.


---------

**Haystack Repositories**

1. "deepset-ai/haystack"

Contains the core code for the Haystack framework and a few components.
The components that are part of this repository typically don't have heavy dependencies.


2. "deepset-ai/haystack-core-integrations"

This is a mono-repo maintained by the deepset-Team that contains integrations for the Haystack framework.
Typically, an integration consists of one or more components. Some integrations only contain document stores.
Each integration is a standalone pypi-package but you can find all of them in the core integrations repo.


3. "deepset-ai/haystack-experimental"

Contains experimental features for the Haystack framework.

"""
