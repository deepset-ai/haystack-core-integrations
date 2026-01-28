# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenSearch Painless scripts used by the document store.

These are Painless (OpenSearch/Elasticsearch's scripting language) scripts,
not Python scripts. They are executed server-side by OpenSearch.
"""

# OpenSearch/Elasticsearch's Painless scripting language script to compute Jaccard similarity for metadata search.
# (n-gram based similarity, n=3 by default)
# Between: a metadata field value (doc[params.field]) a query string (params.q)
METADATA_SEARCH_JACCARD_SCRIPT = """
String a = doc[params.field].size() != 0 ? doc[params.field].value : null;
String b = params.q;
if (a == null || b == null) return 0.0;

a = a.toLowerCase();
b = b.toLowerCase();
int n = params.n;

if (a.length() < n || b.length() < n) {
    return a.equals(b)
        ? 1.0
        : (a.contains(b) || b.contains(a) ? 0.5 : 0.0);
}

Set s1 = new HashSet();
for (int i = 0; i <= a.length() - n; i++) {
    s1.add(a.substring(i, i + n));
}

Set s2 = new HashSet();
for (int i = 0; i <= b.length() - n; i++) {
    s2.add(b.substring(i, i + n));
}

int inter = 0;
for (def x : s1) {
    if (s2.contains(x)) inter++;
}

int union = s1.size() + s2.size() - inter;
if (union == 0) return 0.0;

return inter / (double) union;
"""
