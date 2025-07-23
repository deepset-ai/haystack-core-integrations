---
name: New Integration Proposal
about: Track the creation process for a new integration
title: ''
labels: new integration
assignees: ''

---

## Summary and motivation

Briefly explain the request: why do we need this integration? What use cases does it support?

## Detailed design

Explain the design in enough detail for somebody familiar with Haystack to understand, and for somebody familiar with
the implementation to implement. Get into specifics and corner-cases, and include examples of how the feature is used.
Also, if there's any new terminology involved, define it here.

## Checklist

If the request is accepted, ensure the following checklist is complete before closing this issue.

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
