from typing import Literal

RuleName = Literal[
    "movement",
    "shooting",
    "goal",
    "elimination",
]

RuleParams = {
    "movement": {},
    "shooting": {},
    "goal": {
        "end_episode": ["boolean", False]
    },
    "elimination": {},
}