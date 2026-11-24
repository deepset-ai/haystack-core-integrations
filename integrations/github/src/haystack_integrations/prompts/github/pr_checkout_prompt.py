# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
PR_CHECKOUT_PROMPT = """Check out a pull request's head commit into a local directory and return the repository's \
guidelines from the base branch and the diff of every changed file.

Every diff line starts with its line number in the new file: `+N|` added, ` N|` unchanged, `-|` removed.
Files left out of the diff (lockfiles, binary files, or files over the size limit) are listed at the end. Read them, and
the code around the change, in the checked-out directory.
"""
