import torch

# def NPCCloss(recon, GT, ch_size=1, batch_size=1):
#     [h, w] = recon.shape[2:4]
#     x = recon.view(batch_size, ch_size, h * w)
#     y = GT.view(batch_size, ch_size, h * w)

#     x_mean = torch.mean(x, dim=-1, keepdim=True)
#     y_mean = torch.mean(y, dim=-1, keepdim=True)

#     vx = (x - x_mean)
#     vy = (y - y_mean)

#     c = torch.mean(vx * vy, dim=-1) / (torch.sqrt(torch.mean(vx ** 2, dim=-1) + 1e-08) * torch.sqrt(
#         torch.mean(vy ** 2, dim=-1) + 1e-08))
#     output = torch.mean(-c)  # torch.mean(1-c**2)
#     # output = torch.mean(1-c**2)
#     return output

# # (1, 5, h, w)
# # --> (1, 10, h, w) / (1, 10, h, w)


# Chat GPT version
def NPCCloss(recon, GT, ch_size=1, batch_size=1):
    [h, w] = recon.shape[2:4]
    x = recon.reshape(batch_size, ch_size, h * w)
    y = GT.reshape(batch_size, ch_size, h * w)

    x_mean = torch.mean(x, dim=-1, keepdim=True)
    y_mean = torch.mean(y, dim=-1, keepdim=True)

    vx = x - x_mean
    vy = y - y_mean

    c = torch.mean(vx * vy, dim=-1) / (torch.sqrt(torch.mean(vx.abs() ** 2, dim=-1) + 1e-08) * torch.sqrt(torch.mean(vy.abs() ** 2, dim=-1) + 1e-08))
    output = torch.mean(-c)
    # output = -c
    return output
