from dataclasses import asdict, dataclass
import json


@dataclass
class GraderScore:
	"""Structured grading result for one bot answer."""

	semantic_correctness: int
	helpfulness: int
	tone_safety: int
	passed: bool
	reason: str

	def __post_init__(self) -> None:
		for field_name, value in (
			("semantic_correctness", self.semantic_correctness),
			("helpfulness", self.helpfulness),
			("tone_safety", self.tone_safety),
		):
			if not 0 <= value <= 5:
				raise ValueError(f"{field_name} must be between 0 and 5, got {value}")

		if not self.reason.strip():
			raise ValueError("reason must be a non-empty string")

	def to_dict(self) -> dict[str, int | bool | str]:
		"""Serialize score object for JSON/CSV output."""
		return asdict(self)

	def to_json(self) -> str:
		"""Serialize score object as a JSON string."""
		return json.dumps(self.to_dict(), ensure_ascii=False)
