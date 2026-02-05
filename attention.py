import torch
from typing import Optional

from utils import _get_qkv_projections

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux import FluxAttention,FluxTransformer2DModel,FluxAttnProcessor


# modify from diffusers/models/transformers/transformer_flux.py/FluxAttnProcessor class
class FluxDegAttnProcessor_degconcat:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")
    
    def __call__(
            self,
            attn:"FluxAttention",
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            deg_embeddding:Optional[torch.Tensor]=None,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        query, key, value, encoder_query, encoder_key, encoder_value,deg_query,deg_value,deg_key = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states,deg_embeddding
        )
        # [batch_size, seq_len, inner_dim]->[batch_size,seq_len,heads,head_dim]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply RMSNorm on query and key
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # ----------------------------------------------
        if deg_embeddding is not None:
            deg_query=deg_query.unflatten(-1,(attn.heads,-1))
            deg_key=deg_key.unflatten(-1,(attn.heads,-1))
            deg_value=deg_value.unflatten(-1,(attn.heads,-1))

            query=torch.cat([deg_query,query],dim=1)
            key=torch.cat([deg_key,key],dim=1)
            value=torch.cat([deg_value,value],dim=1)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # [batch_size,seq_len_text+seq_len_image+seq_deg_image,heads,head_dim]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

        

class FluxDegAttnProcessor_deg_add:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")
    
    def __call__(
            self,
            attn:"FluxAttention",
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            deg_embeddding:Optional[torch.Tensor]=None,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        query, key, value, encoder_query, encoder_key, encoder_value,deg_query,deg_value,deg_key = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states,deg_embeddding
        )
        # [batch_size, seq_len, inner_dim]->[batch_size,seq_len,heads,head_dim]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply RMSNorm on query and key
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # ----------------------------------------------
        if deg_embeddding is not None:
            deg_query=deg_query.unflatten(-1,(attn.heads,-1))
            deg_key=deg_key.unflatten(-1,(attn.heads,-1))
            deg_value=deg_value.unflatten(-1,(attn.heads,-1))

            query = deg_query + query
            key = deg_key + key
            value = deg_value + value

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # [batch_size,seq_len_text+seq_len_image+seq_deg_image,heads,head_dim]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

class Flux2DegAttnProcessor:
    pass