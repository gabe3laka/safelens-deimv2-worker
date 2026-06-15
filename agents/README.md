# Agent Scaffolds

These modules expose Agents 1-6 at the artifact path required by the build
brief. They intentionally re-export the canonical implementation from
`agentic_hse.nodes`; there is no second copy to drift.

Typed contracts are defined in `agentic_hse.models`, including the optional
Pydantic AI integration in `agentic_hse.typed_agents`.
