import torch
import numpy as np

shape = (2,3,)

x = [[1, 2, 3], [4, 5, 6]]
x_data = torch.tensor(x)

y_data = torch.rand(shape)

if torch.cuda.is_available():
    x_cuda = x_data.float().to('cuda')
    y_cuda = y_data.to('cuda')
    #print(f"Device tensor is stored on: {x_data.device}")

    
    t1 = torch.mul(x_cuda, y_cuda)
    print(t1)
    
    #x_compute = x.float() @ y.T



#print(f"Tensor Data: {x_data.view(shape)}")
#print(f"Shape of tensor: {x_data.shape}")
#print(f"Datatype of tensor: {x_data.dtype}")
#print(f"Device tensor is stored on: {x_data.device}")

#print("-----------------------------------------------------------------------")

#print(f"Tensor Data: {y_data.view(shape)}")
#print(f"Shape of tensor: {y_data.shape}")
#print(f"Datatype of tensor: {y_data.dtype}")
#print(f"Device tensor is stored on: {y_data.device}")