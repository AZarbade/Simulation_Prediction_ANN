import torch

class RandomForest(torch.nn.Module):
    def __init__(self, n_trees, n_features, n_classes) -> None:
        super(RandomForest, self).__init__()
        self.n_trees = n_trees
        self.trees = torch.nn.ModuleList([
            torch.nn.Linear(n_features, n_classes)
            for i in range(n_trees)
        ])

    def forward(self, logits):
        tree_preds = [tree(logits).unsqueeze(1) for tree in self.trees]
        return torch.mean(torch.cat(tree_preds, dim=1), dim=1)