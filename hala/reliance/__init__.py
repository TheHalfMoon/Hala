"""Official advisory routing component, without model or authority claims."""
from hala.contracts import Decision, Request
from hala.contracts.mock import mock_decision


class HalaReliance:
    def decide(self, request: Request | dict) -> Decision:
        """Recommend a route without lowering security or granting approval."""
        return mock_decision(request)


__all__ = ["HalaReliance"]
