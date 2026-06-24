#!/usr/bin/env python3

import numpy as np


def thomas_tridiag(a, b, c, d):
    """
    Solve tridiagonal system with vectors a (sub, length n), b (diag), c (super) and rhs d.
    a[0] is unused (can be 0). Returns solution x.
    Operates on 1D numpy arrays.
    """
    n = b.size
    ac = a.copy().astype(float)
    bc = b.copy().astype(float)
    cc = c.copy().astype(float)
    dc = d.copy().astype(float)
    # forward eliminate
    for i in range(1, n):
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]
    x = np.empty(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


def harmonic_mean(a, b):
    """Harmonic mean of a and b, handling zeros robustly."""
    eps = 1e-16
    a = np.asarray(a)
    b = np.asarray(b)
    denom = 1.0 / (a + eps) + 1.0 / (b + eps)
    return 2.0 / denom


def adi_diffusion_3d_varD(
    u0_func,
    D_func,
    Lx,
    Ly,
    Lz,
    Nx,
    Ny,
    Nz,
    dt,
    T,
    bc_type="neumann",
    bc_value=0.0,
    source_func=None,
    save_times=None,
):
    """
    3D ADI solver with spatially varying D.
    Parameters:
        u0_func(x,y,z) -> initial field
        D_func(x,y,z) -> diffusion coefficient (>=0)
        Lx,Ly,Lz -> domain sizes
        Nx,Ny,Nz -> number of cells (uniform)
        dt, T -> timestep and final time
        bc_type -> 'dirichlet' or 'neumann' (applies same type on all faces)
        bc_value -> scalar (Dirichlet value or Neumann flux value; for Neumann we assume zero flux if 0.0)
        source_func(x,y,z,t) -> optional source term; if None, zero source
        save_times -> list of times to store snapshots (optional)

    Returns:
        x,y,z arrays (cell centers), snapshots dict time->u array, final u (Nz,Ny,Nx)
    """
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    z = (np.arange(Nz) + 0.5) * dz
    X, Y, Z = np.meshgrid(
        x, y, z, indexing="xy"
    )  # shapes (Ny,Nx,Nz) but we'll reorder later

    # We'll store arrays in shape (Nz, Ny, Nx) for convenience: u[k,j,i]
    X3 = np.empty((Nz, Ny, Nx))
    Y3 = np.empty((Nz, Ny, Nx))
    Z3 = np.empty((Nz, Ny, Nx))
    for k in range(Nz):
        Z3[k, :, :] = z[k]
    for j in range(Ny):
        Y3[:, j, :] = y[j]
    for i in range(Nx):
        X3[:, :, i] = x[i]

    # initial condition
    u = np.zeros((Nz, Ny, Nx), dtype=float)
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                u[k, j, i] = u0_func(x[i], y[j], z[k])

    # precompute D at cell centers
    Dcc = np.zeros_like(u)
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                Dcc[k, j, i] = D_func(x[i], y[j], z[k])

    # precompute face D (harmonic means) arrays:
    # D_x at faces i+1/2: shape (Nz,Ny,Nx-1)
    D_x = harmonic_mean(Dcc[:, :, 0:-1], Dcc[:, :, 1:])
    D_y = harmonic_mean(Dcc[:, 0:-1, :], Dcc[:, 1:, :])
    D_z = harmonic_mean(Dcc[0:-1, :, :], Dcc[1:, :, :])

    nt = int(np.ceil(T / dt))
    rfac_x = dt / (2.0 * dx * dx)
    rfac_y = dt / (2.0 * dy * dy)
    rfac_z = dt / (2.0 * dz * dz)

    snapshots = {}
    if save_times is None:
        save_times = []
    save_set = set([round(t, 10) for t in save_times])

    tcur = 0.0

    # helper function: get ghost value for Dirichlet/Neumann
    def ghost_value(side, interior_val, interior_neighbor_val=None):
        # side is string 'left','right','bottom','top','front','back' but we only pass type
        if bc_type == "dirichlet":
            return bc_value
        else:  # neumann zero-flux implies mirror ghost (zero gradient)
            # ghost = interior_val (zero derivative)
            # if nonzero flux bc_value (flux), ghost should be interior +/- flux*dx ; skipping that variant for brevity
            return interior_val

    fg, ax = plt.subplots(figsize=(4, 4))
    ax.axis("equal")
    im = ax.imshow(u[:, :, 0], origin="lower", extent=[0, Lx, 0, Ly], aspect="auto")
    ax.set_title(f"{tcur:.3f}/{T:.3f}")
    plt.show()

    # Time loop
    for step in range(nt):
        t = tcur

        # ---- STEP 1: X-implicit, explicit in y,z. Solve for u_star (k,j) lines in x ----
        u_star = np.zeros_like(u)
        # For each fixed (k,j) solve tridiagonal in i
        for k in range(Nz):
            for j in range(Ny):
                # build tridiagonal coefficients a,b,c and RHS for i=0..Nx-1
                a = np.zeros(Nx)  # subdiagonal (a[0] unused)
                b = np.zeros(Nx)
                c = np.zeros(Nx)  # superdiagonal (c[-1] unused)
                RHS = np.zeros(Nx)
                for i in range(Nx):
                    # compute laplacian contributions from y and z (explicit)
                    u_ijk = u[k, j, i]
                    # y-neighbors
                    if j == 0:
                        u_jm = ghost_value("bottom", u[k, j, i])
                    else:
                        u_jm = u[k, j - 1, i]
                    if j == Ny - 1:
                        u_jp = ghost_value("top", u[k, j, i])
                    else:
                        u_jp = u[k, j + 1, i]
                    lap_y = (u_jm - 2.0 * u_ijk + u_jp) / (dy * dy)
                    # z-neighbors
                    if k == 0:
                        u_km = ghost_value("front", u[k, j, i])
                    else:
                        u_km = u[k - 1, j, i]
                    if k == Nz - 1:
                        u_kp = ghost_value("back", u[k, j, i])
                    else:
                        u_kp = u[k + 1, j, i]
                    lap_z = (u_km - 2.0 * u_ijk + u_kp) / (dz * dz)

                    s = 0.0 if source_func is None else source_func(x[i], y[j], z[k], t)
                    RHS[i] = u_ijk + 0.5 * dt * (
                        (Dcc[k, j, i] * lap_y) + (Dcc[k, j, i] * lap_z) + s
                    )

                # coefficients: note D_x faces vary with i
                # For interior face at i-1/2 multiply by rfac_x * D_{i-1/2}
                for i in range(Nx):
                    # left face D_{i-1/2}
                    if i == 0:
                        # left boundary: use ghost/BC to create equivalent coefficient; we follow Dirichlet/Neumann via RHS adjust
                        D_imh = D_x[k, j, 0] if Nx > 1 else Dcc[k, j, 0]  # fallback
                    else:
                        D_imh = D_x[k, j, i - 1]
                    if i < Nx - 1:
                        D_iph = D_x[k, j, i]
                    else:
                        D_iph = D_x[k, j, -1] if Nx > 1 else Dcc[k, j, -1]
                    ai = rfac_x * D_imh
                    ci = rfac_x * D_iph
                    a[i] = -ai
                    c[i] = -ci
                    b[i] = 1.0 + ai + ci
                # Adjust RHS for Dirichlet ghosts at boundaries: move known ghost terms to RHS
                if bc_type == "dirichlet":
                    # left boundary ghost value = bc_value; equation i=0 had term -ai * u_{-1} -> move +ai*bc
                    ai0 = rfac_x * (D_x[k, j, 0] if Nx > 1 else Dcc[k, j, 0])
                    RHS[0] += ai0 * bc_value
                    # right boundary
                    aiN = rfac_x * (D_x[k, j, -1] if Nx > 1 else Dcc[k, j, -1])
                    RHS[-1] += aiN * bc_value

                # solve tridiagonal
                u_row = thomas_tridiag(a, b, c, RHS)
                u_star[k, j, :] = u_row

        # ---- STEP 2: Y-implicit, explicit in x,z. Solve for u_dblstar along j ----
        u_dbl = np.zeros_like(u)
        for k in range(Nz):
            for i in range(Nx):
                a = np.zeros(Ny)
                b = np.zeros(Ny)
                c = np.zeros(Ny)
                RHS = np.zeros(Ny)
                for j in range(Ny):
                    u_ijk = u_star[k, j, i]
                    # x-neighbors explicit
                    if i == 0:
                        u_im = ghost_value("left", u_star[k, j, i])
                    else:
                        u_im = (
                            u_star[k, j - 1 + 0, i - 1]
                            if False
                            else u_star[k, j, i - 1]
                        )  # simpler line; j index not used here
                        u_im = (
                            u_star[k, j, i - 1]
                            if i > 0
                            else ghost_value("left", u_star[k, j, i])
                        )
                    if i == Nx - 1:
                        u_ip = ghost_value("right", u_star[k, j, i])
                    else:
                        u_ip = u_star[k, j, i + 1]
                    lap_x = (u_im - 2.0 * u_ijk + u_ip) / (dx * dx)
                    # z-neighbors explicit
                    if k == 0:
                        u_km = ghost_value("front", u_star[k, j, i])
                    else:
                        u_km = u_star[k - 1, j, i]
                    if k == Nz - 1:
                        u_kp = ghost_value("back", u_star[k, j, i])
                    else:
                        u_kp = u_star[k + 1, j, i]
                    lap_z = (u_km - 2.0 * u_ijk + u_kp) / (dz * dz)
                    s = (
                        0.0
                        if source_func is None
                        else source_func(x[i], y[j], z[k], t + 0.5 * dt)
                    )
                    RHS[j] = u_ijk + 0.5 * dt * (
                        (Dcc[k, j, i] * lap_x) + (Dcc[k, j, i] * lap_z) + s
                    )

                # coefficients from D_y faces (varies with j)
                for j in range(Ny):
                    if j == 0:
                        D_jmh = D_y[k, 0, i] if Ny > 1 else Dcc[k, 0, i]
                    else:
                        D_jmh = D_y[k, j - 1, i]
                    if j < Ny - 1:
                        D_jph = D_y[k, j, i]
                    else:
                        D_jph = D_y[k, -1, i] if Ny > 1 else Dcc[k, -1, i]
                    aj = rfac_y * D_jmh
                    cj = rfac_y * D_jph
                    a[j] = -aj
                    c[j] = -cj
                    b[j] = 1.0 + aj + cj

                if bc_type == "dirichlet":
                    RHS[0] += (
                        rfac_y * (D_y[k, 0, i] if Ny > 1 else Dcc[k, 0, i]) * bc_value
                    )
                    RHS[-1] += (
                        rfac_y * (D_y[k, -1, i] if Ny > 1 else Dcc[k, -1, i]) * bc_value
                    )

                col_sol = thomas_tridiag(a, b, c, RHS)
                for j in range(Ny):
                    u_dbl[k, j, i] = col_sol[j]

        # ---- STEP 3: Z-implicit, explicit in x,y. Solve for u^{n+1} along k ----
        u_new = np.zeros_like(u)
        for j in range(Ny):
            for i in range(Nx):
                a = np.zeros(Nz)
                b = np.zeros(Nz)
                c = np.zeros(Nz)
                RHS = np.zeros(Nz)
                for k in range(Nz):
                    u_ijk = u_dbl[k, j, i]
                    # x neighbors explicit
                    if i == 0:
                        u_im = ghost_value("left", u_dbl[k, j, i])
                    else:
                        u_im = u_dbl[k, j, i - 1]
                    if i == Nx - 1:
                        u_ip = ghost_value("right", u_dbl[k, j, i])
                    else:
                        u_ip = u_dbl[k, j, i + 1]
                    lap_x = (u_im - 2.0 * u_ijk + u_ip) / (dx * dx)
                    # y neighbors explicit
                    if j == 0:
                        u_jm = ghost_value("bottom", u_dbl[k, j, i])
                    else:
                        u_jm = u_dbl[k, j - 1, i]
                    if j == Ny - 1:
                        u_jp = ghost_value("top", u_dbl[k, j, i])
                    else:
                        u_jp = u_dbl[k, j + 1, i]
                    lap_y = (u_jm - 2.0 * u_ijk + u_jp) / (dy * dy)
                    s = (
                        0.0
                        if source_func is None
                        else source_func(x[i], y[j], z[k], t + dt)
                    )
                    RHS[k] = u_ijk + 0.5 * dt * (
                        (Dcc[k, j, i] * lap_x) + (Dcc[k, j, i] * lap_y) + s
                    )

                for k in range(Nz):
                    if k == 0:
                        D_kmh = D_z[0, j, i] if Nz > 1 else Dcc[0, j, i]
                    else:
                        D_kmh = D_z[k - 1, j, i]
                    if k < Nz - 1:
                        D_kph = D_z[k, j, i]
                    else:
                        D_kph = D_z[-1, j, i] if Nz > 1 else Dcc[-1, j, i]
                    aik = rfac_z * D_kmh
                    cik = rfac_z * D_kph
                    a[k] = -aik
                    c[k] = -cik
                    b[k] = 1.0 + aik + cik

                if bc_type == "dirichlet":
                    RHS[0] += (
                        rfac_z * (D_z[0, j, i] if Nz > 1 else Dcc[0, j, i]) * bc_value
                    )
                    RHS[-1] += (
                        rfac_z * (D_z[-1, j, i] if Nz > 1 else Dcc[-1, j, i]) * bc_value
                    )

                col_sol = thomas_tridiag(a, b, c, RHS)
                for k in range(Nz):
                    u_new[k, j, i] = col_sol[k]

        tcur += dt
        u = u_new

        if int(tcur // dt) % 10 == 0:
            print(f"{tcur:.3f}/{T:.3f}")
            fg, ax = plt.subplots(figsize=(4, 4))
            ax.axis("equal")
            im = ax.imshow(
                u[:, :, 0], origin="lower", extent=[0, Lx, 0, Ly], aspect="auto"
            )
            ax.set_title(f"{tcur:.3f}/{T:.3f}")
            plt.show()
            # fg.close()
            # snapshots[round(tcur,10)] = u.copy()

    return x, y, z, snapshots, u


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # domain
    Lx = Ly = Lz = 1.0
    Nx = Ny = Nz = 32  # keep moderate for example
    dt = 1e-3
    T = 0.1

    # initial condition: gaussian centered at 0.5,0.5,0.5
    def u0(x, y, z):
        x0, y0, z0 = 0.5, 0.5, 0.5
        sig = 0.04
        return np.exp(
            -((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sig * sig)
        )

    def Dfunc(x, y, z):
        return 0.05 if (x < 0.5 and y < 0.5) else 0.1

    # zero source
    source = None

    # Neumann BC (zero flux) implemented by mirroring interior value; pass bc_type='neumann'
    xc, yc, zc, snaps, u_final = adi_diffusion_3d_varD(
        u0,
        Dfunc,
        Lx,
        Ly,
        Lz,
        Nx,
        Ny,
        Nz,
        dt,
        T,
        bc_type="neumann",
        bc_value=0.0,
        source_func=source,
        save_times=[T],
    )

    # write final result slices to CSV (z-slices)
    import csv

    for k, zv in enumerate(zc):
        fname = f"u_final_z{k:03d}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            for j in range(Ny):
                writer.writerow(u_final[k, j, :].tolist())
    print("Wrote final z-slices to CSV.")
    print("max u:", np.max(u_final), "min u:", np.min(u_final))
