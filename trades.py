import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.031,
                epsilon=0.007,
                perturb_steps=5,
                beta=1.0,
                distance='l_inf'):
    """
    TRADES-style loss with optional support for unlabeled samples (y == -1).

    - Natural CE loss is computed only on labeled samples.
    - Robust KL loss is computed on all samples.
    """
    del optimizer  # kept for backward compatibility with existing callers

    perturb_steps = int(perturb_steps)
    if perturb_steps <= 0:
        raise ValueError("perturb_steps must be positive")

    device = x_natural.device
    batch_size = x_natural.size(0)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    model.eval()

    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural, device=device)

    if distance == "l_inf":
        for _ in range(perturb_steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif distance == "l_2":
        delta = 0.001 * torch.randn_like(x_natural, device=device).detach()
        delta = Variable(delta.data, requires_grad=True)

        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1.0) * criterion_kl(
                    F.log_softmax(model(adv), dim=1),
                    F.softmax(model(x_natural), dim=1),
                )
            loss.backward()

            grad_view = delta.grad.view(batch_size, -1)
            grad_norms = grad_view.norm(p=2, dim=1)
            zero_mask = grad_norms == 0
            if zero_mask.any():
                delta.grad[zero_mask] = torch.randn_like(delta.grad[zero_mask])
                grad_view = delta.grad.view(batch_size, -1)
                grad_norms = grad_view.norm(p=2, dim=1)

            grad_norms = torch.clamp(grad_norms, min=1e-12)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            optimizer_delta.step()

            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)

        x_adv = (x_natural + delta).detach()

    else:
        raise ValueError(f"Unknown distance: {distance!r}")

    model.train()

    logits = model(x_natural)
    logits_adv = model(x_adv)

    loss_robust = criterion_kl(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits, dim=1),
    )

    if y is None:
        loss_natural = torch.tensor(0.0, device=device)
    else:
        labeled_mask = y >= 0
        if torch.any(labeled_mask):
            loss_natural = F.cross_entropy(logits[labeled_mask], y[labeled_mask])
        else:
            loss_natural = torch.tensor(0.0, device=device)

    return loss_natural + beta * loss_robust
