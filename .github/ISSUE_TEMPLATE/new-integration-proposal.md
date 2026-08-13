---
name: New Integration Proposal
about: Track the creation process for a new integration
title: ''
labels: new integration
assignees: ''

---

## Summary and motivation

Briefly explain the request: why do we need this integration? What use cases does it support?

## Adoption signals

Every integration we merge has to be maintained, tested nightly, and released, so we prioritize integrations for
technologies with a healthy and growing user base. Please share whatever signals you can find for the software you
want to integrate with — you don't need all of them, but the more you provide, the easier it is for us to decide.

- **GitHub stars** of the main repository (and roughly how fast it is growing):
- **PyPI downloads in the last 30 days** of the Python client/SDK (see [pypistats.org](https://pypistats.org/)):
- **Release activity**: date of the latest release and rough release cadence:
- **Maintenance**: is the project actively maintained (recent commits, issues being answered)? Is there a company or
  foundation behind it?
- **Haystack community demand**: links to Discord threads, GitHub issues, or other requests asking for this
  integration:
- **Anything else** that shows adoption (enterprise usage, conference talks, blog posts, comparable integrations in
  other frameworks):

> Example: [`elasticsearch-py`](https://github.com/elastic/elasticsearch-py) has 4.4k GitHub stars and
> [54 million PyPI downloads in the last 30 days](https://pypistats.org/packages/elasticsearch), with releases every
> few weeks and Elastic maintaining it.

If the technology is very new and these numbers are still small, tell us why you expect it to grow — that is useful
context too.

## Detailed design

Explain the design in enough detail for somebody familiar with Haystack to understand, and for somebody familiar with
the implementation to implement. Get into specifics and corner-cases, and include examples of how the feature is used.
Also, if there's any new terminology involved, define it here.

## Checklist

If the request is accepted, ensure the following checklist is complete before closing this issue.
Follow the instructions in https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md#create-a-new-integration and use our scaffolding script for the implementation.

### Tasks
- [ ] The code is documented with docstrings and was merged in the `main` branch
- [ ] Docs are published at https://docs.haystack.deepset.ai/
- [ ] There is a Github workflow running the tests for the integration nightly and at every PR
- [ ] A new label named like `integration:<your integration name>` has been added to the list of labels for this [repository](https://github.com/deepset-ai/haystack-core-integrations/labels)
- [ ] The [labeler.yml](https://github.com/deepset-ai/haystack-core-integrations/blob/main/.github/labeler.yml) file has been updated
- [ ] The package has been released on PyPI
- [ ] An integration tile with a usage example has been added to https://github.com/deepset-ai/haystack-integrations
- [ ] The integration has been listed in the [Inventory section](https://github.com/deepset-ai/haystack-core-integrations#inventory) of this repo README
- [ ] The feature was announced through social media
