"""Optimizer-correctness contract identifiers shared by the optimizer and the interpreter."""

# v3 adds context measurement v2, bound-field native views and chunked plans.
# Created only while the context-chunking flag is on. It keeps v2 measurement.
CONTEXT_CHUNKING_CONTRACT = "optimizer-correctness-v3"
CONDITIONAL_CONTRACT = "optimizer-correctness-v5"
# v6 adds evaluation-semantics-v2 (value equality, the metric registry and
# direction-aware objectives). It freezes the task-metric and conditional
# capability booleans instead of implying conditional creation.
EVALUATION_FIDELITY_CONTRACT = "optimizer-correctness-v6"
SPAN_GROUNDING_CONTRACT = "optimizer-correctness-v7"
# Learned-step requests delimit the compiler-added instruction and stage inputs
# in one developer message (``instruction_envelope.py``). Frozen per run and per
# revision; ``None`` keeps the legacy request bytes. No setting creates it yet.
INSTRUCTION_ENVELOPE_CONTRACT = "instruction-envelope-v1"
