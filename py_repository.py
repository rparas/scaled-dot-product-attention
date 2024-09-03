import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calculates the attention weights.
    Q, K, V must have matching leading dimensions.
    K, V must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    Q: query shape == (..., seq_len_q, depth)
    K: key shape == (..., seq_len_k, depth)
    V: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = np.dot(Q, K.transpose())  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = K.shape[-1]  # dimension of the key
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is applied to the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = np.exp(scaled_attention_logits)
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    output = np.dot(attention_weights, V)  # (..., seq_len_q, depth_v)

    return output, attention_weights

# Example usage:
np.random.seed(0)
Q = np.random.rand(1, 3, 64)  # Query
K = np.random.rand(1, 3, 64)  # Key
V = np.random.rand(1, 3, 64)  # Value
output, attention_weights = scaled_dot_product_attention(Q, K, V)
print("Output:", output)
print("Attention Weights:", attention_weights)
