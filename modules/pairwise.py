import torch


def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))

    return x1,x2


def Class2Simi(x,mode='cls',mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1)==n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2    
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

if __name__ == '__main__':
    # Pair Enum
    # Create a 2D tensor
    x = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # Create a mask
    mask = torch.tensor([True, False, True])

    # Call PairEnum function
    x1, x2 = PairEnum(x)

    print("x1:", x1)
    print("x2:", x2)

    # x1: tensor([[1, 2],                    x2: tensor([[1, 2],
    #     [3, 4],                                   [1, 2],
    #     [5, 6],                                   [1, 2],
    #     [1, 2],                                   [3, 4],                              
    #     [3, 4],                                   [3, 4],
    #     [5, 6],                                   [3, 4],
    #     [1, 2],                                   [5, 6],
    #     [3, 4],                                   [5, 6],
    #     [5, 6]])                                  [5, 6]])
    #
    #     So, x1 and x2 contain all possible pairs of the input tensor x

    # Create a 1D tensor
    x = torch.tensor([1, 2, 3, 2, 1])

    # Call Class2Simi function
    out = Class2Simi(x, mode='cls')

    print("Output:", out)

    # Output: tensor( [1, 0, 0, 0, 1,       # first element is comapared with all other elements
    #                 0, 1, 0, 1, 0,        # second element is compared with all other elements
    #                 0, 0, 1, 0, 0,        # third element is compared with all other elements
    #                 0, 1, 0, 1, 0,        .....
    #                 1, 0, 0, 0, 1] )      .....
     