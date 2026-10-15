"""Official planning component with a weight-free deterministic mock."""
from hala.contracts import Plan, Request
from hala.contracts.mock import mock_plan


class HalaMind:
    def plan(self, request: Request | dict) -> Plan:
        """Return bounded planning advice; never execute a plan."""
        return mock_plan(request)


__all__ = ["HalaMind"]
