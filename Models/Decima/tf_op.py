import numpy as np

import torch
from torch.nn.parameter import Parameter
cuda = torch.cuda.is_available()

cuda=False
print("cuda", cuda)

def expand_act_on_state(state, sub_acts):
    # expand the state by explicitly adding in actions
    batch_size = state.size()[0]
    num_nodes = state.size()[1]
    num_features = state.shape[2]  # deterministic
    expand_dim = len(sub_acts)

    # replicate the state
    state = torch.tile(state, [1, 1, expand_dim])
    state = torch.reshape(state,
        [batch_size, num_nodes * expand_dim, num_features])


    # prepare the appended sub-actions
    sub_acts = torch.tensor(sub_acts, dtype=torch.float32)
    sub_acts = torch.reshape(sub_acts, [1, 1, expand_dim])
    sub_acts = torch.tile(sub_acts, [1, 1, num_nodes])
    sub_acts = torch.reshape(sub_acts, [1, num_nodes * expand_dim, 1])
    sub_acts = torch.tile(sub_acts, [batch_size, 1, 1])
    if cuda:
        sub_acts = sub_acts.cuda()

    # concatenate expanded state with sub-action features
    concat_state = torch.concat([state, sub_acts], dim=2)

    return concat_state


def glorot(shape):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    init_range = torch.sqrt(torch.tensor(6.0) / torch.tensor(shape[0] + shape[1]))
    init = Parameter(torch.FloatTensor(shape))
    init = init.clone()
    init.uniform_(-init_range)
    return init


'''
def leaky_relu(features, alpha=0.2, name=None):
  """Compute the Leaky ReLU activation function.
  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
  Args:
    features: A `Tensor` representing preactivation values.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)
'''


def masked_outer_product(a, b, mask):
    """
    combine two probability distribution together
    a: batch_size * num_nodes
    b: batch_size * (num_executor_limit * num_jobs)
    """
    batch_size = a.size()[0]
    num_nodes = a.size()[1]
    num_limits = b.size()[1]

    a = torch.reshape(a, [batch_size, num_nodes, 1])
    b = torch.reshape(b, [batch_size, 1, num_limits])

    # outer matrix product
    outer_product = a * b
    outer_product = torch.reshape(outer_product, [batch_size, -1])

    # mask
    outer_product = torch.transpose(outer_product)
    outer_product = torch.boolean_mask(outer_product, mask)
    outer_product = torch.transpose(outer_product)

    return outer_product


