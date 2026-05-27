from .jax_agent import JaxDistributionalFlowRL
from .jax_value_flow_agent import JaxValueFlowRL

AGENT_REGISTRY = {
    # JAX λ-flow agent (your original one)
    "jax_lambda_flow": JaxDistributionalFlowRL.create,

    # JAX value-flow agent with DCFM-style loss
    "jax_value_flow": JaxValueFlowRL.create,
}

def make_agent(name: str, **kwargs):
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY.keys())}")
    factory = AGENT_REGISTRY[name]
    return factory(**kwargs)