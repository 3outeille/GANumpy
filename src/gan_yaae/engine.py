import numpy as np

class Node:
    
    def __init__(self, data, requires_grad=True, children=[]):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.requires_grad = requires_grad
        self.children = children
        self.grad = None
        if self.requires_grad: 
            self.zero_grad()
        # Stores function.
        self._compute_derivatives = lambda: None

    def __repr__(self):
        return f"Node(data={self.data},\n grad={self.grad})\n"

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = Node(np.zeros_like(self.data, dtype=np.float64), requires_grad=False)

    def backward(self, grad=None):
        L, visited = [], set()
        topo = topo_sort(self, L, visited)
        if grad is None:
            if self.shape == ():
                self.grad = Node(1, requires_grad=False)
            else:
                raise RuntimeError('backward: grad must be specified for non-0 tensor')
        else:
            self.grad = Node(grad, requires_grad=False) if isinstance(grad, (np.ndarray, list)) else grad

        for v in reversed(topo):
            v._compute_derivatives()

    # Operators.
    def __add__(self, other):
        op = Add(self, other)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __radd__(self, other): 
        # other + self.
        op = Add(other, self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out
    
    def sum(self, axis=None, keepdims=True):
        op = Sum(self, axis, keepdims)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __mul__(self, other):
        op = Mul(self, other)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __rmul__(self, other): 
        # other * self.
        op = Mul(other, self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out
    
    def matmul(self, other):
        op = Matmul(self, other)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __neg__(self): 
        # -self.
        op = Neg(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __sub__(self, other):
        # self - other.
        op = Add(self, -other)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __rsub__(self, other):
        # other - self.
        op = Add(other, -self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __truediv__(self, other): 
        op = Div(self, other)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def __rtruediv__(self, other): 
        # other / self.
        op = Div(other, self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out
    
    # Functions.
    def tanh(self):
        op = Tanh(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out
    
    def relu(self):
        op = ReLU(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def lrelu(self, alpha=0.):
        op = LeakyReLU(self, alpha)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out    

    def exp(self):
        op = Exp(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def log(self):
        op = Log(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out
    
    def sigmoid(self):
        op = Sigmoid(self)
        out = op.forward_pass()
        out._compute_derivatives = op.compute_derivatives
        return out

    def dropout(self, p=0.2): 
            op = Dropout(self, p)
            out = op.forward_pass()
            out._compute_derivatives = op.compute_derivatives
            return out

# ----------------------------------------------------------------------------
# Operators & Functions.

class Add():
    
    def __init__(self, node1, node2):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)
    
    def forward_pass(self):
        self.out = Node(np.add(self.node1.data, self.node2.data), children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        if self.node1.requires_grad:
            self.node1.grad.data += compress_gradient(self.out.grad.data, self.node1.shape)
        if self.node2.requires_grad:
            self.node2.grad.data += compress_gradient(self.out.grad.data, self.node2.shape)

class Sum():

    def __init__(self, node1, axis=None, keepdims=True):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.axis = axis
        self.keepdims = keepdims

    def forward_pass(self):
        self.out = Node(np.sum(self.node1.data, axis=self.axis, keepdims=self.keepdims), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        if self.axis != None and self.keepdims == False:
            self.node1.grad.data += np.expand_dims(self.out.grad.data, self.axis) * np.ones_like(self.node1.data)
        else:
            self.node1.grad.data += self.out.grad.data * np.ones_like(self.node1.data)

class Mul():

    def __init__(self, node1, node2):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)
    
    def forward_pass(self):
        self.out = Node(np.multiply(self.node1.data, self.node2.data), children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        if self.node1.requires_grad:
            self.node1.grad.data += compress_gradient(self.out.grad.data, self.node2.shape) * self.node2.data
        if self.node2.requires_grad:
            self.node2.grad.data += compress_gradient(self.out.grad.data, self.node1.shape) * self.node1.data

class Matmul():

    def __init__(self, node1, node2):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)

    def forward_pass(self):
        self.out = Node(self.node1.data @ self.node2.data, children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        dim = [i for i in range(len(self.node1.data.shape))]
        if len(dim) > 1:
            dim[-1], dim[-2] = dim[-2], dim[-1]

        if self.node1.requires_grad:
            self.node1.grad.data = self.out.grad.data @ self.node2.data.transpose(dim)
        if self.node2.requires_grad:
            self.node2.grad.data = self.node1.data.transpose(dim) @ self.out.grad.data

class Neg():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
   
    def forward_pass(self):
        self.out = Node(-self.node1.data, children=[self.node1])
        return self.out

    def compute_derivatives(self):
        if self.node1.requires_grad:
            self.node1.grad.data += -self.out.grad.data * np.ones_like(self.node1.data)

class Div():

    def __init__(self, node1, node2):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.node2 = node2 if isinstance(node2, Node) else Node(node2)
    
    def forward_pass(self):
        self.out = Node(np.divide(self.node1.data, self.node2.data), children=[self.node1, self.node2])
        return self.out

    def compute_derivatives(self):
        if self.node1.requires_grad:
            self.node1.grad.data += compress_gradient(self.out.grad.data, self.node1.shape) * 1/(self.node2.data)
        if self.node2.requires_grad:
            self.node2.grad.data += compress_gradient(self.out.grad.data, self.node2.shape) * -self.node1.data/(self.node2.data)**2

class Tanh():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
    
    def forward_pass(self):
        self.out = Node(np.tanh(self.node1.data), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad.data += self.out.grad.data * (1 - np.tanh(self.node1.data)**2)
        
class ReLU():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)

    def forward_pass(self):
        self.out = Node(self.node1.data * (self.node1.data > 0), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad.data += self.out.grad.data * (self.node1.data > 0)

class LeakyReLU():

    def __init__(self, node1, alpha):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.alpha = alpha

    def forward_pass(self):
        X = self.node1.data 
        self.out = Node(np.where(X > 0, X, X * self.alpha), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        dX = np.ones_like(self.node1.data)
        dX[self.node1.data < 0] = self.alpha
        self.node1.grad.data += self.out.grad.data * dX

class Exp():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)

    def forward_pass(self):
        self.out = Node(np.exp(self.node1.data), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad.data += self.out.grad.data * np.exp(self.node1.data)

class Log():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)

    def forward_pass(self):
        self.out = Node(np.log(self.node1.data), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad.data += self.out.grad.data * (1/self.node1.data)

class Sigmoid():

    def __init__(self, node1):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)

    def forward_pass(self):
        self.out = Node(1. / (1. + np.exp(-self.node1.data)), children=[self.node1])
        return self.out

    def compute_derivatives(self):
        f = 1. / (1. + np.exp(-self.node1.data))
        self.node1.grad.data += self.out.grad.data * f * (1 - f)

class Dropout():

    def __init__(self, node1, p):
        self.node1 = node1 if isinstance(node1, Node) else Node(node1)
        self.p = p
        self.mask = np.random.uniform(size=node1.shape) > self.p

    def forward_pass(self):
        self.out = Node(self.node1.data * self.mask, children=[self.node1])
        return self.out

    def compute_derivatives(self):
        self.node1.grad.data += self.out.grad.data * self.mask

# ----------------------------------------------------------------------------
# Utility functions.

def topo_sort(v, L, visited):
    """
        Returns list of all of the children in the graph in topological order.

        Parameters:
        - v: last node in the graph.
        - L: list of all of the children in the graph in topological order (empty at the beginning).
        - visited: set of visited children.
    """
    if v not in visited:
        visited.add(v)
        for child in v.children:
            topo_sort(child, L, visited)
        L.append(v)
    return L

def compress_gradient(grad, other_tensor_shape):
    """
        Returns the gradient but compressed (needed when gradient shape mismatch during reverse mode).

        Paramaters:
        - grad: gradient.
        - other_tensor_shape: shape of target tensor.
    """
    ndims_added = grad.ndim - len(other_tensor_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(other_tensor_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad