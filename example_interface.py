# Make sure to disable weight_decay on A_proj!

from quick_scan.scan_interface import scan

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    Example for using the scan interface
'''
# from quick_ssm.layers import SSM

# class Model(nn.Module):
#     def __init__(
#             self,
#             num_layers=2,
#             hidden_size=512,
#             state_size_mult=4.0,
#             eps=1e-5,
#             dtype=torch.float32,
#             device=None,
#             **kwargs
#         ):  
#         super().__init__()
#         self.layers = []

#         for _ in range(num_layers):
#             ssm_layer = SSM(
#                     hidden_size=hidden_size,
#                     state_size_mult=state_size_mult,
#                     eps=eps,
#                     dtype=dtype,
#                     device=device,
#                     **kwargs
#                 )
#             self.layers.append(ssm_layer)
#         self.layers = nn.ModuleList(self.layers)
#         # )

#     def forward(self, x):
#         # x: Input tensor of shape (B, L, D)
#         # B is the batch size
#         # L is the sequence length
#         # D is the hidden dimension

#         for layer in self.layers:
#             x = layer(x)
#         return x

# if __name__ == "__main__":
#     model = Model(
#         num_layers=2,
#         hidden_size=128,
#         state_size_mult=4.0,
#     )
#     print(model)

#     random_data = torch.randn(2, 1024, 512).cuda()