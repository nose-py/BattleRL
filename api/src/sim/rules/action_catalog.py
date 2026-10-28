from collections.abc import Sequence

from src.sim.rules.actions import ACTION_NAMES
from src.sim.rules.base_rule import BaseRule


class ActionCatalog:
    def __init__(self, rules: Sequence[BaseRule]):
        declared: dict[int, str] = {}

        for rule in rules:
            for code, name in [(c, ACTION_NAMES.get(c)) for c in rule.declared_actions()]:

                if type(code) is not int or code < 0:
                    raise ValueError(
                        f"Invalid action code: {code!r}"
                    )

                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"Invalid action name: {name!r}"
                    )

                declared[code] = name

        if not declared:
            raise ValueError(
                "At least one enabled rule must declare an action"
            )

        self.codes = tuple(sorted(declared))

        self.names = tuple(
            declared[code]
            for code in self.codes
        )

        self._indices = {
            code: index
            for index, code in enumerate(self.codes)
        }

    @property
    def n_actions(self) -> int:
        return len(self.codes)

    def decode(self, index: int) -> int:
        """
        Policy index -> internal action code.
        """
        if (
            type(index) is not int
            or not 0 <= index < self.n_actions
        ):
            raise ValueError(
                f"Invalid action index: {index!r}"
            )

        return self.codes[index]

    def index_for_code(self, code: int) -> int:
        """
        Internal action code -> policy index.
        """
        return self._indices[code]

    def describe(self) -> list[dict]:
        return [
            {
                "index": index,
                "code": code,
                "name": name,
            }
            for index, (code, name) in enumerate(
                zip(self.codes, self.names)
            )
        ]