import torch
import torch.nn.functional as nf


def compute_layer_pv_projection(last_token_hidden_states, layers_pv_vector, projection_type="proj"):
    """Helper function."""
    layers_pv_projection = []
    if len(last_token_hidden_states) != len(layers_pv_vector):
        raise ValueError(
            "Hidden-state and PV-vector layer counts must match: "
            f"hidden_states={len(last_token_hidden_states)}, "
            f"pv_vectors={len(layers_pv_vector)}"
        )
    for i, layer_hidden in enumerate(last_token_hidden_states):
        if projection_type == "proj":
            pv_projection = compute_projection(
                activation=layer_hidden.squeeze(),  
                pv_vector=layers_pv_vector[i].to(layer_hidden.device)
            )
        elif projection_type == "cos":
            pv_projection = nf.cosine_similarity(
                layer_hidden,  
                layers_pv_vector[i].to(layer_hidden.device)
            ).item()
        else:
            raise TypeError("projection_type must be 'proj' or 'cos'")
        layers_pv_projection.append(pv_projection)

    return layers_pv_projection


def compute_last_hidden_diff(benign_hidden_states, pv_hidden_states):
    # Use PV minus benign consistently so the vector points toward policy violation.
    benign_hidden = benign_hidden_states[:, -1, :].squeeze().cpu()
    pv_hidden = pv_hidden_states[:, -1, :].squeeze().cpu()
    layers_diff = pv_hidden - benign_hidden
    return layers_diff


def compute_projection(activation, pv_vector):
    
    # Normalize only the PV vector to retain the activation's signed projection.
    normalized_vector = torch.nn.functional.normalize(pv_vector, p=2, dim=0)
    normalized_vector = normalized_vector.to(device=activation.device, dtype=activation.dtype)
    
    projection = torch.dot(activation, normalized_vector)
    return projection.item()  
