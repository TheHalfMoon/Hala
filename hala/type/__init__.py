"""Official grounding component with a weight-free deterministic mock."""
from hala.contracts import Proposal, Request
from hala.contracts.mock import mock_action


class HalaType:
    def action(self, request: Request | dict) -> Proposal:
        """Return an observation-bound proposal for independent local policy."""
        return mock_action(request)


__all__ = ["HalaType"]
