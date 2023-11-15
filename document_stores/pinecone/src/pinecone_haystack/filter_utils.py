import logging
from typing import Union, List, Dict
from abc import ABC, abstractmethod
from collections import defaultdict
from haystack.preview.errors import FilterError
from pinecone_haystack.errors import PineconeDocumentStoreFilterError

logger = logging.getLogger(__file__)


def nested_defaultdict() -> defaultdict:
    """
    Data structure that recursively adds a dictionary as value if a key does not exist. Advantage: In nested dictionary
    structures, we don't need to check if a key already exists (which can become hard to maintain in nested dictionaries
    with many levels) but access the existing value if a key exists and create an empty dictionary if a key does not
    exist.
    """
    return defaultdict(nested_defaultdict)


class LogicalFilterClause(ABC):
    """
    Class that is able to parse a filter and convert it to the format that the underlying databases of our
    DocumentStores require.

    Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
    operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`, `"$gte"`, `"$lt"`,
    `"$lte"`) or a metadata field name.
    Logical operator keys take a dictionary of metadata field names and/or logical operators as
    value. Metadata field names take a dictionary of comparison operators as value. Comparison
    operator keys take a single value or (in case of `"$in"`) a list of values as value.
    If no logical operator is provided, `"$and"` is used as default operation. If no comparison
    operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
    operation.
    Example:
        ```python
        filters = {
            "$and": {
                "type": {"$eq": "article"},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
                "$or": {
                    "genre": {"$in": ["economy", "politics"]},
                    "publisher": {"$eq": "nytimes"}
                }
            }
        }
        # or simpler using default operators
        filters = {
            "type": "article",
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": ["economy", "politics"],
                "publisher": "nytimes"
            }
        }
        ```

    To use the same logical operator multiple times on the same level, logical operators take optionally a list of
    dictionaries as value.

    Example:
        ```python
        filters = {
            "$or": [
                {
                    "$and": {
                        "Type": "News Paper",
                        "Date": {
                            "$lt": "2019-01-01"
                        }
                    }
                },
                {
                    "$and": {
                        "Type": "Blog Post",
                        "Date": {
                            "$gte": "2019-01-01"
                        }
                    }
                }
            ]
        }
        ```

    """

    def __init__(self, conditions: List[Union["LogicalFilterClause", "ComparisonOperation"]]):
        self.conditions = conditions

    @abstractmethod
    def evaluate(self, fields) -> bool:
        pass

    @classmethod
    def parse(cls, filter_term: Union[dict, List[dict]]) -> Union["LogicalFilterClause", "ComparisonOperation"]:
        """
        Parses a filter dictionary/list and returns a LogicalFilterClause instance.

        :param filter_term: Dictionary or list that contains the filter definition.
        """
        conditions: List[Union[LogicalFilterClause, ComparisonOperation]] = []

        if isinstance(filter_term, dict):
            filter_term = [filter_term]
        for item in filter_term:
            for key, value in item.items():
                if key == "$not":
                    conditions.append(NotOperation.parse(value))
                elif key == "$and":
                    conditions.append(AndOperation.parse(value))
                elif key == "$or":
                    conditions.append(OrOperation.parse(value))
                # Key needs to be a metadata field
                else:
                    conditions.extend(ComparisonOperation.parse(key, value))

        if cls == LogicalFilterClause:
            if len(conditions) == 1:
                return conditions[0]
            else:
                return AndOperation(conditions)
        else:
            return cls(conditions)

    def convert_to_pinecone(self):
        """
        Converts the LogicalFilterClause instance to a Pinecone filter.
        """
        pass


class ComparisonOperation(ABC):
    def __init__(self, field_name: str, comparison_value: Union[str, int, float, bool, List]):
        self.field_name = field_name
        self.comparison_value = comparison_value

    @abstractmethod
    def evaluate(self, fields) -> bool:
        pass

    @classmethod
    def parse(cls, field_name, comparison_clause: Union[Dict, List, str, float]) -> List["ComparisonOperation"]:
        comparison_operations: List[ComparisonOperation] = []

        if isinstance(comparison_clause, dict):
            for comparison_operation, comparison_value in comparison_clause.items():
                if comparison_operation == "$eq":
                    comparison_operations.append(EqOperation(field_name, comparison_value))
                elif comparison_operation == "$in":
                    comparison_operations.append(InOperation(field_name, comparison_value))
                elif comparison_operation == "$ne":
                    comparison_operations.append(NeOperation(field_name, comparison_value))
                elif comparison_operation == "$nin":
                    comparison_operations.append(NinOperation(field_name, comparison_value))
                elif comparison_operation == "$gt":
                    comparison_operations.append(GtOperation(field_name, comparison_value))
                elif comparison_operation == "$gte":
                    comparison_operations.append(GteOperation(field_name, comparison_value))
                elif comparison_operation == "$lt":
                    comparison_operations.append(LtOperation(field_name, comparison_value))
                elif comparison_operation == "$lte":
                    comparison_operations.append(LteOperation(field_name, comparison_value))

        # No comparison operator is given, so we use the default operators "$in" if the comparison value is a list and
        # "$eq" in every other case
        elif isinstance(comparison_clause, list):
            comparison_operations.append(InOperation(field_name, comparison_clause))
        else:
            comparison_operations.append((EqOperation(field_name, comparison_clause)))

        return comparison_operations

    def convert_to_pinecone(self):
        """
        Converts the ComparisonOperation instance to a Pinecone comparison operator.
        """
        pass

    def invert(self) -> "ComparisonOperation":
        """
        Inverts the ComparisonOperation.
        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.
        (https://github.com/semi-technologies/weaviate/issues/1717)
        """
        pass


class NotOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'NOT' operations.
    """

    def evaluate(self, fields) -> bool:
        return not any(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_pinecone(self) -> Dict[str, Union[str, int, float, bool, List[Dict]]]:
        conditions = [condition.invert().convert_to_pinecone() for condition in self.conditions]
        if len(conditions) > 1:
            # Conditions in self.conditions are by default combined with AND which becomes OR according to DeMorgan
            return {"$or": conditions}
        else:
            return conditions[0]

    def invert(self) -> Union[LogicalFilterClause, ComparisonOperation]:
        # This method is called when a "$not" operation is embedded in another "$not" operation. Therefore, we don't
        # invert the operations here, as two "$not" operation annihilate each other.
        # (If we have more than one condition, we return an AndOperation, the default logical operation for combining
        # multiple conditions.)
        if len(self.conditions) > 1:
            return AndOperation(self.conditions)
        else:
            return self.conditions[0]


class AndOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'AND' operations.
    """

    def evaluate(self, fields) -> bool:
        return all(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_pinecone(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_pinecone() for condition in self.conditions]
        return {"$and": conditions}

    def invert(self) -> "OrOperation":
        return OrOperation([condition.invert() for condition in self.conditions])


class OrOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'OR' operations.
    """

    def evaluate(self, fields) -> bool:
        return any(condition.evaluate(fields) for condition in self.conditions)

    def convert_to_pinecone(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_pinecone() for condition in self.conditions]
        return {"$or": conditions}

    def invert(self) -> AndOperation:
        return AndOperation([condition.invert() for condition in self.conditions])


class EqOperation(ComparisonOperation):
    """
    Handles conversion of the '$eq' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] == self.comparison_value

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[List[str], str, int, float, bool]]]:
        return {self.field_name: {"$eq": self.comparison_value}}

    def invert(self) -> "NeOperation":
        return NeOperation(self.field_name, self.comparison_value)


class InOperation(ComparisonOperation):
    """
    Handles conversion of the '$in' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False

        if not isinstance(self.comparison_value, list):
            raise PineconeDocumentStoreFilterError("'$in' operation requires comparison value to be a list.")

        # If the document field is a list, check if any of its values are in the comparison value
        if isinstance(fields[self.field_name], list):
            return any(field in self.comparison_value for field in fields[self.field_name])

        return fields[self.field_name] in self.comparison_value

    def convert_to_pinecone(self) -> Dict[str, Dict[str, List]]:
        if not isinstance(self.comparison_value, list):
            raise PineconeDocumentStoreFilterError("'$in' operation requires comparison value to be a list.")
        return {self.field_name: {"$in": self.comparison_value}}

    def invert(self) -> "NinOperation":
        return NinOperation(self.field_name, self.comparison_value)


class NeOperation(ComparisonOperation):
    """
    Handles conversion of the '$ne' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] != self.comparison_value

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[List[str], str, int, float, bool]]]:
        return {self.field_name: {"$ne": self.comparison_value}}

    def invert(self) -> "EqOperation":
        return EqOperation(self.field_name, self.comparison_value)


class NinOperation(ComparisonOperation):
    """
    Handles conversion of the '$nin' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return True

        if not isinstance(self.comparison_value, list):
            raise PineconeDocumentStoreFilterError("'$nin' operation requires comparison value to be a list.")

        # If the document field is a list, check if any of its values are in the comparison value
        if isinstance(fields[self.field_name], list):
            return not any(field in self.comparison_value for field in fields[self.field_name])

        return fields[self.field_name] not in self.comparison_value

    def convert_to_pinecone(self) -> Dict[str, Dict[str, List]]:
        if not isinstance(self.comparison_value, list):
            raise PineconeDocumentStoreFilterError("'$in' operation requires comparison value to be a list.")
        return {self.field_name: {"$nin": self.comparison_value}}

    def invert(self) -> "InOperation":
        return InOperation(self.field_name, self.comparison_value)


class GtOperation(ComparisonOperation):
    """
    Handles conversion of the '$gt' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False

        # If the document field is a list, check if any of its values are greater than the comparison value
        if isinstance(fields[self.field_name], list):
            return any(field > self.comparison_value for field in fields[self.field_name])

        return fields[self.field_name] > self.comparison_value

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if not isinstance(self.comparison_value, (float, int)):
            raise PineconeDocumentStoreFilterError("Comparison value for '$gt' operation must be a float or int.")
        return {self.field_name: {"$gt": self.comparison_value}}

    def invert(self) -> "LteOperation":
        return LteOperation(self.field_name, self.comparison_value)


class GteOperation(ComparisonOperation):
    """
    Handles conversion of the '$gte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False

        # If the document field is a list, check if any of its values are greater than or equal to the comparison value
        if isinstance(fields[self.field_name], list):
            return any(field >= self.comparison_value for field in fields[self.field_name])

        return fields[self.field_name] >= self.comparison_value

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if not isinstance(self.comparison_value, (float, int)):
            raise PineconeDocumentStoreFilterError("Comparison value for '$gte' operation must be a float or int.")
        return {self.field_name: {"$gte": self.comparison_value}}

    def invert(self) -> "LtOperation":
        return LtOperation(self.field_name, self.comparison_value)


class LtOperation(ComparisonOperation):
    """
    Handles conversion of the '$lt' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False

        # If the document field is a list, check if any of its values are less than the comparison value
        if isinstance(fields[self.field_name], list):
            return any(field < self.comparison_value for field in fields[self.field_name])

        return fields[self.field_name] < self.comparison_value

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if not isinstance(self.comparison_value, (float, int)):
            raise PineconeDocumentStoreFilterError("Comparison value for '$lt' operation must be a float or int.")
        return {self.field_name: {"$lt": self.comparison_value}}

    def invert(self) -> "GteOperation":
        return GteOperation(self.field_name, self.comparison_value)


class LteOperation(ComparisonOperation):
    """
    Handles conversion of the '$lte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False

        # If the document field is a list, check if any of its values are less than or equal to the comparison value
        if isinstance(fields[self.field_name], list):
            return any(field <= self.comparison_value for field in fields[self.field_name])

        return fields[self.field_name] <= self.comparison_value

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if not isinstance(self.comparison_value, (float, int)):
            raise PineconeDocumentStoreFilterError("Comparison value for '$lte' operation must be a float or int.")
        return {self.field_name: {"$lte": self.comparison_value}}

    def invert(self) -> "GtOperation":
        return GtOperation(self.field_name, self.comparison_value)
