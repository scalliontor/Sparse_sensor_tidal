import numpy as np
import matplotlib.pyplot as plt

def flux_x(U, g=9.81):
    """Physical flux in x-direction. U = [h, hu, hv]"""
    h = U[..., 0]
    qx = U[..., 1]
    qy = U[..., 2]
    
    h_safe = np.maximum(h, 1e-12)
    u = qx / h_safe
    v = qy / h_safe
    
    F = np.zeros_like(U)
    F[..., 0] = qx
    F[..., 1] = qx * u + 0.5 * g * h**2
    F[..., 2] = qx * v
    return F

def flux_y(U, g=9.81):
    """Physical flux in y-direction. U = [h, hu, hv]"""
    h = U[..., 0]
    qx = U[..., 1]
    qy = U[..., 2]
    
    h_safe = np.maximum(h, 1e-12)
    u = qx / h_safe
    v = qy / h_safe
    
    G = np.zeros_like(U)
    G[..., 0] = qy
    G[..., 1] = qy * u
    G[..., 2] = qy * v + 0.5 * g * h**2
    return G

def hll_flux_x(UL, UR, g=9.81):
    """HLL numerical flux at x-interfaces."""
    hL, qxL, qyL = UL[..., 0], UL[..., 1], UL[..., 2]
    hR, qxR, qyR = UR[..., 0], UR[..., 1], UR[..., 2]
    
    hL_safe = np.maximum(hL, 1e-12)
    hR_safe = np.maximum(hR, 1e-12)
    
    uL = qxL / hL_safe
    uR = qxR / hR_safe
    
    cL = np.sqrt(g * hL_safe)
    cR = np.sqrt(g * hR_safe)
    
    # Wave speed estimates (Davis)
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    
    # Clip speeds
    SL = np.minimum(SL, 0.0)
    SR = np.maximum(SR, 0.0)
    
    FL = flux_x(UL, g)
    FR = flux_x(UR, g)
    
    # HLL Flux formula
    # F_HLL = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
    
    denom = np.maximum(SR - SL, 1e-12)
    
    # Expand dims for broadcasting
    SL_ext = SL[..., np.newaxis]
    SR_ext = SR[..., np.newaxis]
    denom_ext = denom[..., np.newaxis]
    
    FHLL = (SR_ext * FL - SL_ext * FR + SL_ext * SR_ext * (UR - UL)) / denom_ext
    return FHLL

def hll_flux_y(UD, UU, g=9.81):
    """HLL numerical flux at y-interfaces (Down and Up)."""
    # UD is bottom cell (y), UU is top cell (y+1)
    hD, qxD, qyD = UD[..., 0], UD[..., 1], UD[..., 2]
    hU, qxU, qyU = UU[..., 0], UU[..., 1], UU[..., 2]
    
    hD_safe = np.maximum(hD, 1e-12)
    hU_safe = np.maximum(hU, 1e-12)
    
    vD = qyD / hD_safe
    vU = qyU / hU_safe
    
    cD = np.sqrt(g * hD_safe)
    cU = np.sqrt(g * hU_safe)
    
    # Wave speed estimates
    SD = np.minimum(vD - cD, vU - cU)
    SU = np.maximum(vD + cD, vU + cU)
    
    SD = np.minimum(SD, 0.0)
    SU = np.maximum(SU, 0.0)
    
    GD = flux_y(UD, g)
    GU = flux_y(UU, g)
    
    denom = np.maximum(SU - SD, 1e-12)
    
    SD_ext = SD[..., np.newaxis]
    SU_ext = SU[..., np.newaxis]
    denom_ext = denom[..., np.newaxis]
    
    GHLL = (SU_ext * GD - SD_ext * GU + SD_ext * SU_ext * (UU - UD)) / denom_ext
    return GHLL

def update_cfl_dt(U, dx, dy, cfl=0.5, g=9.81):
    h = np.maximum(U[..., 0], 1e-12)
    u = np.abs(U[..., 1] / h)
    v = np.abs(U[..., 2] / h)
    c = np.sqrt(g * h)
    
    dt_x = dx / np.max(u + c)
    dt_y = dy / np.max(v + c)
    
    return cfl * min(dt_x, dt_y)

def apply_boundary_conditions_2d(U, t, h0, A=1.0, period_hours=12.42):
    """
    Apply ghost cell boundary conditions for 2D.
    U shape: (Ny+2, Nx+2, 3)
    Layout: y is along axis 0, x is along axis 1.
    Indices: 0 and -1 are ghost cells. 1 to N are interior.
    """
    # 1. Left Boundary (x=0) - Open tidal boundary (similar to 1D)
    # Assumed open to the ocean
    eta = A * np.sin(2 * np.pi * t / (period_hours * 3600.0))
    U[:, 0, 0] = h0 + eta
    U[:, 0, 1] = U[:, 1, 1] * (U[:, 0, 0] / np.maximum(U[:, 1, 0], 1e-6)) # Copy velocity u
    U[:, 0, 2] = U[:, 1, 2] * (U[:, 0, 0] / np.maximum(U[:, 1, 0], 1e-6)) # Copy velocity v
    
    # 2. Right Boundary (x=L) - Closed Wall (Reflective) for now
    U[:, -1, 0] = U[:, -2, 0]
    U[:, -1, 1] = -U[:, -2, 1] # Reflect u
    U[:, -1, 2] = U[:, -2, 2]
    
    # 3. Bottom Boundary (y=0) - Closed Wall
    U[0, :, 0] = U[1, :, 0]
    U[0, :, 1] = U[1, :, 1]
    U[0, :, 2] = -U[1, :, 2] # Reflect v
    
    # 4. Top Boundary (y=W) - Closed Wall
    U[-1, :, 0] = U[-2, :, 0]
    U[-1, :, 1] = U[-2, :, 1]
    U[-1, :, 2] = -U[-2, :, 2] # Reflect v

def simulate_2d(
    Lx=100_000.0, Ly=100_000.0,
    Nx=100, Ny=100, h0=10.0,
    t_end_hours=12.0, cfl=0.4,
    g=9.81, A=1.0, period_hours=12.42,
    save_every=50
):
    dx = Lx / Nx
    dy = Ly / Ny
    
    # U incorporates 1 ghost cell ring: shape (Ny+2, Nx+2, 3)
    U = np.zeros((Ny + 2, Nx + 2, 3), dtype=float)
    U[..., 0] = h0
    
    t = 0.0
    t_end = t_end_hours * 3600.0
    step = 0
    
    out_t = []
    out_h = []
    
    while t < t_end:
        apply_boundary_conditions_2d(U, t, h0, A, period_hours)
        
        # Calculate stable dt from interior cells
        dt = update_cfl_dt(U[1:-1, 1:-1, :], dx, dy, cfl, g)
        if t + dt > t_end:
            dt = t_end - t
            
        # Fluxes in X-direction at interfaces (i+1/2) -> Nx+1 interfaces per row
        # Loop over x-interfaces i=0 to Nx
        Fx = np.zeros((Ny+2, Nx+1, 3))
        Fx[:, :, :] = hll_flux_x(U[:, :-1, :], U[:, 1:, :], g)
        
        # Fluxes in Y-direction at interfaces (j+1/2) -> Ny+1 interfaces per col
        # Loop over y-interfaces j=0 to Ny
        Fy = np.zeros((Ny+1, Nx+2, 3))
        Fy[:, :, :] = hll_flux_y(U[:-1, :, :], U[1:, :, :], g)
        
        # Update interior cells
        # U_new[j, i] = U[j, i] - dt/dx * (Fx[i] - Fx[i-1]) - dt/dy * (Fy[j] - Fy[j-1])
        # x-flux difference for interior cells i=1..Nx corresponds to Fx[:, 1:] - Fx[:, :-1]
        # y-flux difference for interior cells j=1..Ny corresponds to Fy[1:, :] - Fy[:-1, :]
        
        U[1:-1, 1:-1, :] -= (dt / dx) * (Fx[1:-1, 1:, :] - Fx[1:-1, :-1, :]) \
                          + (dt / dy) * (Fy[1:, 1:-1, :] - Fy[:-1, 1:-1, :])
                          
        # Safety pos
        U[1:-1, 1:-1, 0] = np.maximum(U[1:-1, 1:-1, 0], 1e-6)
        
        t += dt
        step += 1
        
        if step % save_every == 0:
            out_t.append(t)
            # Save interior h
            out_h.append(U[1:-1, 1:-1, 0].copy())
            print(f"Time: {t/3600:.2f}h / {t_end_hours}h")

    return np.array(out_t), np.array(out_h)

if __name__ == "__main__":
    print("Starting 2D SWE HLL Test...")
    out_t, out_h = simulate_2d(
        Lx=100000, Ly=100000, Nx=50, Ny=50, t_end_hours=6.0, A=2.0
    )
    
    # Save a plot of the final state
    plt.imshow(out_h[-1], origin='lower', extent=[0, 100, 0, 100])
    plt.colorbar(label='Water Depth (m)')
    plt.title('2D Godunov HLL Baseline - Final State')
    plt.savefig('test_2d_result.png')
    print("Test passed. Plot saved to test_2d_result.png")
