from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ScoreCard:
    """Collects one song's weighted score components and the reasons behind them."""
    weights: Dict[str, float]
    components: List[Tuple[float, float]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def add(self, key: str, label: str, similarity: float, detail: str) -> None:
        """Record one weighted component and a reason showing its exact point contribution."""
        weight = self.weights.get(key, 0.0)
        if weight <= 0:
            return
        self.components.append((weight, similarity))
        self.reasons.append(f"{label}: {detail} (+{weight * similarity:.3f} of {weight:.2f})")