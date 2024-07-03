import torch
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')


def get_sim_matrix_slow(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity matrix between feature vectors in a sure but slow manner.

    Args:
        features (torch.Tensor): A tensor of size (ns, nt + 1, feature_dim). +1 is from anchor

    Returns:
        torch.Tensor: A similarity matrix of size (ns, nt + 1, nt + 1).

    """

    ns, ntp1, feature_dim = features.size()
    similarity_matrices = torch.zeros(ns, ntp1, ntp1, dtype=torch.float32)
    for i in range(ns):
        current_features = features[i]
        similarity_matrix = F.cosine_similarity(current_features.unsqueeze(1), current_features.unsqueeze(0), dim=2)
        similarity_matrices[i] = similarity_matrix
    return similarity_matrices


def get_sim_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity matrix between feature vectors efficiently.

    Args:
        features (torch.Tensor): A tensor of size (ns, nt + 1, feature_dim).

    Returns:
        torch.Tensor: A similarity matrix of size (ns, nt + 1, nt + 1).
    """

    similarity_matrix = F.cosine_similarity(features.unsqueeze(2), features.unsqueeze(1), dim=3)
    return similarity_matrix


def get_loss_simple(features, labels, tau=0.1):
    """
       Compute the loss, a modified supervised contrastive learning loss.

       Args:
           features (torch.Tensor): A tensor of size (ns, nt + 1, feature_dim).
           labels (torch.Tensor): A tensor of size (ns, nt + 1) with anchor labels (0), positive labels (1),
                                  and negative labels (2).
           tau (float): Temperature parameter for scaling the similarity matrix.

       Returns:
           torch.Tensor: The computed loss value.

       Raises:
           ValueError: If the labels do not meet the expected format or if there is more than one anchor (0) in labels
           for the same unique sample.

       """

    # i is the anchor index, only one zero inside labels
    if torch.any(torch.sum(labels == 0, dim=1) != 1).item():
        raise ValueError("More than one zero (anchor) found in labels for each unique sample.")

    if features.shape[0:2] != labels.shape:
        raise ValueError("Number of features and number of labels do not match")

    sim_mat = (get_sim_matrix(features) / tau).float()
    exp_sim_mat = torch.exp(sim_mat)
    anchor_indices = torch.nonzero(labels == 0)

    pos_mask = -F.normalize((labels == 1).float(), p=1, dim=1)  # includes the -1/|P(i)| in the mask
    pos_mask = pos_mask.contiguous()
    numerator = torch.einsum('ij, ij -> i', sim_mat[anchor_indices[:, 0], anchor_indices[:, 1], :], pos_mask)

    not_anchor_mask = (labels != 0).float().contiguous()
    denominator = torch.einsum('ij, ij -> i', exp_sim_mat[anchor_indices[:, 0], anchor_indices[:, 1], :], not_anchor_mask)
    denominator = torch.log(denominator)
    return torch.mean(numerator + denominator)


if __name__ == '__main__':
    label2name = ['anchor', 'positive', 'negative']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tau = 0.1
    labels = torch.tensor([[0, 1, 2],
                           [0, 1, 2],
                           [0, 1, 2]])
    labels = labels.contiguous()
    features = torch.tensor([[[13, 1, -1, -1], [11, 1, -1, 1], [-1, 12, 1, 1]],
                             [[13, 1, -1, -1], [11, 1, -1, 1], [-1, 12, 1, 1]],
                             [[13, 1, -1, -1], [11, 1, -1, 1], [-1, 12, 1, 1]]], dtype=torch.float)
    features = features.contiguous()
    loss = get_loss_simple(features, labels, tau)
