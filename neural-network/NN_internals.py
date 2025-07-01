import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm

"""
MAIN CONFIGURATION PARAMETERS (TWEAK THESE)
"""
# Domain parameters
U_RANGE = (-.1, .4)  # Range for u coordinate
V_RANGE = (-.2, .3)   # Range for v coordinate
RESOLUTION = 100        # Grid resolution for plotting

# Fixed point coordinates (theory values)
u0 = 343 / (288 * np.pi)
v0 = -49 / (288 * np.pi)

# Linear regime boundary condition parameters
BC_RADIUS = 10**-2        # Radius around fixed point where linear BC is enforced
BC_WEIGHT = 10**-4         # Weight for boundary condition loss term

# DOF limiting parameters
DOF_WEIGHT = 0

# Neural network architecture
HIDDEN_LAYERS = [50, 100, 100, 50]  # Neurons per hidden layer
ACTIVATION = nn.Tanh()    # Activation function

# Training parameters
LEARNING_RATE = 1e-4
# BATCH_SIZE = 100

# Numerical offset to avoid singularities and zeros
NUM_OFFSET = 1e-10

"""
NEURAL NETWORK DEFINITION
"""
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        layers = []
        # Input layer (2 features: u and v)
        layers.append(nn.Linear(2, HIDDEN_LAYERS[0]))
        layers.append(ACTIVATION)
        
        # Hidden layers (customizable depth and width)
        for i in range(len(HIDDEN_LAYERS)-1):
            layers.append(nn.Linear(HIDDEN_LAYERS[i], HIDDEN_LAYERS[i+1]))
            layers.append(ACTIVATION)
        
        # Output layer (1 value: F(u,v))
        layers.append(nn.Linear(HIDDEN_LAYERS[-1], 1))
        
        self.net = nn.Sequential(*layers)
        self.register_buffer('fixed_point', torch.tensor([[u0, v0]], dtype=torch.float32))

    def forward(self, x):
        # Compute current network output at fixed point
        fp_output = self.net(self.fixed_point)
        # Apply dynamic shift
        return self.net(x) - fp_output

"""
PHYSICS FUNCTIONS
"""
def beta_u(u, v):
    """Beta function for u coordinate"""
    return 2 * u - (3 * u**3) / (2 * torch.pi * (v + u)**3) + NUM_OFFSET

def beta_v(u, v):
    """Beta function for v coordinate"""
    return -(u**2 * (7 * v + u)) / (4 * torch.pi * (v + u)**3) + NUM_OFFSET

def F_star(u, v):
    """Linearized solution around fixed point (boundary condition)"""
    sqrt_43 = torch.sqrt(torch.tensor(43.0))
    term1 = -(172 - 137 * sqrt_43) * u
    term2 = (44 * sqrt_43 + 215) * v
    term3 = -49 * (305 * sqrt_43 - 473) / (96 * torch.pi)
    return term1 + term2 + term3

"""
TRAINING FUNCTIONS
"""
def sample_boundary_points(num_points):
    """Sample points near fixed point for BC enforcement"""
    noise = torch.rand(num_points, 2) * 2 - 1  # Uniform in [-1,1]
    noise = noise / torch.norm(noise, dim=1, keepdim=True) * BC_RADIUS * torch.rand(num_points, 1)
    return torch.tensor([u0, v0]) + noise

def compute_loss(net, u, v, x_bc, epoch):
    """Compute combined PDE and boundary condition loss"""
    # PDE loss at collocation points
    x = torch.stack((u, v), dim=-1).requires_grad_(True)
    F = net(x)
    dF = torch.autograd.grad(F, x, create_graph=True, grad_outputs=torch.ones_like(F))[0]
    pde_residual = beta_u(u, v) * dF[:, 0] + beta_v(u, v) * dF[:, 1]
    pde_loss = torch.mean(pde_residual**2)
    
    # Boundary condition loss (linear regime)
    F_bc_pred = net(x_bc)
    F_bc_true = F_star(x_bc[:, 0], x_bc[:, 1]).unsqueeze(-1)
    bc_loss = torch.mean((F_bc_pred - F_bc_true)**2)
    
    # DOF loss at collocation points (force ||dF|| = 1)
    dof_residual = (dF[:, 0]**2 + dF[:, 1]**2)**0.5 - 1
    dof_loss = torch.mean(dof_residual**2)
    
    true_bc_weight = BC_WEIGHT
    true_dof_weight = DOF_WEIGHT
    true_pde_weight = 1 - true_bc_weight - true_dof_weight
    
    return true_pde_weight * pde_loss + true_bc_weight * bc_loss + true_dof_weight * dof_loss

def train_model(EPOCHS, print_every, BATCH_SIZE):
    """Train the Physics-Informed Neural Network"""
    net = PINN()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Sample collocation points (avoid v = -u singularity and u=0 and v=0)
        u = torch.rand(BATCH_SIZE, 1) * (U_RANGE[1] - U_RANGE[0]) + U_RANGE[0]
        v = torch.rand(BATCH_SIZE, 1) * (V_RANGE[1] - V_RANGE[0]) + V_RANGE[0]
        mask = (u + v).abs() > 0.1
        u, v = u[mask], v[mask]
        mask = u.abs() > NUM_OFFSET
        u, v = u[mask], v[mask]
        mask = v.abs() > NUM_OFFSET
        u, v = u[mask], v[mask]
        
        # Sample boundary points
        x_bc = sample_boundary_points(BATCH_SIZE//2)
        
        # Compute and backpropagate loss
        loss = compute_loss(net, u, v, x_bc, epoch)
        loss.backward()
        optimizer.step()
        
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}")
    
    return net

"""
VISUALIZATION FUNCTIONS
"""
def find_nn_solution(net):
    """Find where F(0,v)=0 for neural network solution"""
    v_line = np.linspace(V_RANGE[0], V_RANGE[1], 500)
    points = torch.tensor(np.column_stack((np.zeros_like(v_line), v_line))).float()
    with torch.no_grad():
        F_line = net(points).numpy().flatten()
    
    # Find zero crossing
    sign_change = np.diff(np.sign(F_line))
    zero_crossings = np.where(sign_change)[0]
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        v_nn = (v_line[idx] + v_line[idx+1])/2
        print(f"Neural network solution at v = {v_nn:.3f}")
        return v_nn
    else:
        print("Warning: No zero crossing found on u=0 line")
        return None

def plot_results(net, name):
    """Generate publication-quality plot of results"""
    # Create evaluation grid
    u = np.linspace(U_RANGE[0], U_RANGE[1], RESOLUTION)
    v = np.linspace(V_RANGE[0], V_RANGE[1], RESOLUTION)
    U, V = np.meshgrid(u, v)
    
    # Prepare grid points tensor (with correct bracket structure)
    points = torch.tensor(np.column_stack((U.ravel(), V.ravel()))).float()
    
    with torch.no_grad():
        F = net(points).numpy().reshape(U.shape)
    
    # Find neural network solution
    v_nn = find_nn_solution(net)
    
    # Calculate beta function flow
    beta_u_vals = beta_u(torch.tensor(U), torch.tensor(V)).numpy()
    beta_v_vals = beta_v(torch.tensor(U), torch.tensor(V)).numpy()
    magnitude = np.sqrt(beta_u_vals**2 + beta_v_vals**2)
    
    # Normalize vectors for visualization
    norm = np.sqrt(beta_u_vals**2 + beta_v_vals**2)
    U_norm = beta_u_vals / (norm + 1e-8)
    V_norm = beta_v_vals / (norm + 1e-8)
    colors = plt.cm.gray((magnitude - magnitude.min())/(magnitude.max() - magnitude.min()))
    
    # Calculate symmetric limits centered at 0
    max_abs = np.max(np.abs(F))
    vmin, vmax = -max_abs, max_abs
    
    # Create normalization centered at 0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap with centered colors
    cmap = plt.cm.RdBu_r  # Blue-white-red reversed
    contour = ax.contourf(U, V, F, levels=50, cmap=cmap, norm=norm, alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('F(u,v)', rotation=270, labelpad=20)
    
    # Plot beta function flow (every 5th vector)
    skip = 5
    for i in range(0, U.shape[0], skip):
        for j in range(0, U.shape[1], skip):
            ax.quiver(U[i,j], V[i,j], U_norm[i,j], V_norm[i,j],
                     color=colors[i,j], scale=25, width=0.003,
                     alpha=0.7, pivot='tail')
    
    # Plot zero level set
    ax.contour(U, V, F, levels=[0], colors='k', linestyles=':', linewidths=1.5)
    
    # Reference elements
    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='u=0 (k=0)')
    ax.axhline(0.15, color='k', linestyle='-.', linewidth=1, label='correct result (v=0.15)')
    ax.scatter(u0, v0, s=200, marker='*', color='k', edgecolor='k', label='NGFP')
    
    # Plot neural network solution if found
    if v_nn is not None:
        ax.scatter(0, v_nn, s=100, marker='o', color='r', 
                  label=f'PINN result (v={v_nn:.3f})')
    
    # Formatting
    ax.set_xlim(U_RANGE)
    ax.set_ylim(V_RANGE)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title('UV-critical surface heightmap')
    
    # Custom legend
    z0_proxy = Line2D([0], [0], color='k', linestyle=':', linewidth=1.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(z0_proxy)
    labels.append('UV-critical surface (F=0)')
    ax.legend(handles, labels, loc='upper right')
    
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
    plt.show()

"""
MAIN EXECUTION
"""
if __name__ == "__main__":
    # Train and plot the solution
    pinn = train_model()
    plot_results(pinn)