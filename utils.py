import torch.nn 
from diffusers.models import FluxAttention

def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None,deg_embedding=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    deg_query,deg_key,deg_value = (None,)
    if deg_embedding is not None:
        deg_query=attn.to_q_deg(hidden_states)
        deg_key=attn.to_k_deg(hidden_states)
        deg_value=attn.to_v_deg(hidden_states)
    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value,deg_query,deg_key,deg_value


def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None,deg_embedding=None):# -> tuple[Any, Any, Any, Any | tuple[None], Any | tuple[None]...:
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    
    deg_query,deg_key,deg_value=(None,)
    if deg_embedding is not None:
        deg_query,deg_key,deg_value=attn.to_qkv_deg(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value, deg_query,deg_key,deg_value


def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None,deg_embedding=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states,deg_embedding)
    return _get_projections(attn, hidden_states, encoder_hidden_states,deg_embedding)
