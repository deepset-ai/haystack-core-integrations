---
name: New Integration Proposal
about: Track the creation process for a new integration
title: ''
labels: New integration request
assignees: ''

---

## Summary and motivation

Briefly explain the feature request: why do we need this feature? What use cases does it support?

## Alternatives

A clear and concise description of any alternative solutions or features you've considered.

## Detailed design

Explain the design in enough detail for somebody familiar with Haystack to understand, and for somebody familiar with the implementation to implement. Get into specifics and corner-cases, and include examples of how the feature is used. Also, if there's any new terminology involved, define it here.

## Checklist

If the feature request is accepted, ensure the following checklist is complete before closing the issue.

- [ ] The package has been released on PyPI
- [ ] There is a Github workflow running the tests for the integration nightly and at every PR
- [ ] A label named like `integration:<your integration name>` has been added to this repo
- [ ] The [labeler.yml](https://github.com/deepset-ai/haystack-core-integrations/blob/main/.github/labeler.yml) file has been updated
- [ ] An integration tile has been added to https://github.com/deepset-ai/haystack-integrations
- [ ] The integration has been listed in the [Inventory section](https://github.com/deepset-ai/haystack-core-integrations#inventory) of this repo README
