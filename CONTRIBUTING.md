# Contributing to Haystack Core Integrations

First off, thanks for taking the time to contribute! :blue_heart:

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents)
for different ways to help and details about how this project handles them. Please make sure to read
the relevant section before making your contribution. It will make it a lot easier for us maintainers
and smooth out the experience for all involved. The community looks forward to your contributions!

> [!TIP]
> If you like Haystack, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star this repository
> - Tweet about it
> - Mention Haystack at local meetups and tell your friends/colleagues

## Table of Contents

- [Contributing to Haystack Core Integrations](#contributing-to-haystack-core-integrations)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [I Have a Question](#i-have-a-question)
  - [Reporting Bugs](#reporting-bugs)
    - [Before Submitting a Bug Report](#before-submitting-a-bug-report)
    - [How Do I Submit a Good Bug Report?](#how-do-i-submit-a-good-bug-report)
  - [Suggesting Enhancements](#suggesting-enhancements)
    - [Before Submitting an Enhancement](#before-submitting-an-enhancement)
    - [How Do I Submit a Good Enhancement Suggestion?](#how-do-i-submit-a-good-enhancement-suggestion)
  - [Contribute code](#contribute-code)
    - [Where to start](#where-to-start)
    - [Setting up your development environment](#setting-up-your-development-environment)
    - [Clone the git repository](#clone-the-git-repository)
    - [Working on integrations](#working-on-integrations)
      - [Integration version](#integration-version)
      - [Run the linter](#run-the-linter)
      - [Run the tests](#run-the-tests)
      - [Create a new integration](#create-a-new-integration)
    - [Improving The Documentation](#improving-the-documentation)
      - [Python API docs](#python-api-docs)
      - [Documentation pages](#documentation-pages)


## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior to haystack@deepset.ai.


## I Have a Question

> [!TIP]
> If you want to ask a question, we assume that you have read the available [documentation](https://docs.haystack.deepset.ai/docs/intro).

Before you ask a question, it is best to search for existing [issues](/../../issues) that might help you. In case you have
found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to
search the internet for answers first.

If you then still feel the need to ask a question and need clarification, you can use one of our
[community channels](https://haystack.deepset.ai/community). Discord in particular is often very helpful.

## Reporting Bugs


### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to
investigate carefully, collect information and describe the issue in detail in your report. Please complete the
following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://docs.haystack.deepset.ai/docs/intro). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](/../../issues?labels=bug).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of Haystack and the integrations you're using
  - Possibly your input and the output
  - If you can reliably reproduce the issue, a snippet of code we can use


### How Do I Submit a Good Bug Report?

> [!IMPORTANT]
> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be reported using [this link](https://github.com/deepset-ai/haystack-core-integrations/security/advisories/new).


We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [issue of type Bug Report](/../../issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=).
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps.
- If the team can reproduce the issue, it will either be scheduled for a fix or made available for [community contribution](#contribute-code).


## Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including new integrations and improvements
to existing ones. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.


### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://docs.haystack.deepset.ai/docs/intro) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](/../../issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing and distributing the integration on your own.


### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as GitHub issues of type [Feature request for existing integrations](/../../issues/new?assignees=&labels=feature+request&projects=&template=feature-request-for-existing-integrations.md&title=).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Fill the issue following the template

## Contribute code

> [!IMPORTANT]
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.


### Where to start

If this is your first contribution, a good starting point is looking for an open issue that's marked with the label
["good first issue"](https://github.com/deepset-ai/haystack-core-integrations/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
The core contributors periodically mark certain issues as good for first-time contributors. Those issues are usually
limited in scope, easy fixable and low priority, so there is absolutely no reason why you should not try fixing them.
It's also a good excuse to start looking into the project and a safe space for experimenting failure: if you don't get the
grasp of something, pick another one!

### Setting up your development environment

Haystack makes heavy use of [Hatch](https://hatch.pypa.io/latest/), a Python project manager that we use to set up the
virtual environments, build the project and publish packages. As you can imagine, the first step towards becoming a
Haystack contributor is installing Hatch. There are a variety of install methods depending on your operating system
platform, version and personal taste: please have a look at [this page](https://hatch.pypa.io/latest/install/#installation)
and keep reading once you can run from your terminal:

```console
$ hatch --version
Hatch, version 1.9.3
```

### Clone the git repository

You won't be able to make changes directly to this repo, so the first step is to [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). Once your fork is ready, you can
clone a local copy with:

```console
$ git clone https://github.com/YOUR-USERNAME/haystack-core-integrations
```

If everything worked, you should be able to do something like this (the output might be different):

```console
$ cd haystack-core-integrations/integrations/chroma
$ hatch version
0.13.1.dev30+g9fe18d23
```

### Working on integrations

> [!NOTE]
> This is a so-called monorepo, meaning that this single repository contains multiple different applications. In our case,
> Haystack Integrations live in the `./integrations` folder, and each one must be considered an independent project.
> All the examples shown in this section must be executed in one specific integration folder.

Hatch provides several commands we use for developing integrations, let's pick `haystack-chroma` to show a practical
example. Before proceeding,  from the root of the repository we `cd` into the integration folder:

```console
$ cd integrations/chroma
```

#### Integration version

We can get the current version of the integration with:

```console
$ hatch version
0.13.1.dev30+g9fe18d23
```

> [!NOTE]
> Even when you see the version number ends with a suffix like `.dev...`, this doesn't necessarily mean the integration
> has unreleased commits. In other words, despite what the version number says, you might be looking at the latest
> released code. This is a side effect of working with a monorepo: hatch can't tell if a certain commit in the git
> history is releated to one integration or the other.

#### Run the linter

Every time you change the code, it's a good practice to format the code and perform linting (with automatic fixes):

```console
$ hatch run fmt
```

To check for static type errors, run:

```console
$ hatch run test:types
```

#### Run the tests

It's important your tests pass before contributing code. To run all the tests locally:

```console
$ hatch run test:all
```

> [!IMPORTANT] The command above will run ALL the tests, including integration tests; some of those often need you to
> run a certain service in background (e.g. a Vector Database) or provide credentials to external services (e.g. OpenAI)
> in order to pass.

To run the unit tests only, run:

```console
$ hatch run test:unit
```

For integration tests, run:

```console
$ hatch run test:integration
```

#### Create a new integration

> [!IMPORTANT]
> Core integrations follow the naming convention `PREFIX-haystack`, where `PREFIX` can be the name of the technology
> you're integrating Haystack with. For example, a deepset integration would be named as `deepset-haystack`.

To create a new integration, from the root of the repo change directory into `integrations`:

```sh
cd integrations
```

From there, use `hatch` to create the scaffold of the new integration:

```sh
$ hatch --config hatch.toml new -i
Project name: deepset-haystack
Description []: An example integration, this text can be edited later

deepset-haystack
├── src
│   └── deepset_haystack
│       ├── __about__.py
│       └── __init__.py
├── tests
│   └── __init__.py
├── LICENSE.txt
├── README.md
└── pyproject.toml
```

### Improving The Documentation

There are two types of documentation for this project: Python API docs, and Documentation pages

#### Python API docs

The Python API docs detail the source code: classes, functions, and parameters that every integration implements.
This type of documentation is extracted from the source code itself, and contributors should pay attention when they
change the code to also change relevant comments and docstrings. This type of documentation is mostly useful to
developers, but it can be handy for users at times. You can browse it on the dedicated section in the
[documentation website](https://docs.haystack.deepset.ai/reference/integrations-chroma).

We use [`haystack-pydoc-tools`](https://github.com/deepset-ai/haystack-pydoc-tools) to convert docstrings into properly formatted Markdown files, and while the CI takes care of
generating and publishing the updated documentation once a new version is released, you can generate the docs
locally using Hatch. From an integration folder:

```console
$ hatch run docs
```

If you want to customise the conversion process, the `haystack-pydoc-tools` config files are stored in a `pydoc/` folder
for each integration.

#### Documentation pages

Documentation pages explain what an integration accomplishes, how to use it, and how to configure it. This
type of documentation mostly targets end-users, and contributors are welcome to make changes and keep it up-to-date.
You can contribute changes to the documentation using the "Suggest Edits" button in the top-right corner of each page.
