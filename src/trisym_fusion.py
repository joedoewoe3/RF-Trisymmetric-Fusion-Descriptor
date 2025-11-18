Pythonimport numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R  # For future SO(3) rotations

# Configurable params
DIM = 3  # 2D or 3D
ARRAY_RADIUS = 1.0  # meters
NOISE_STD = {'tdoa': 1e-9, 'aoa_az': np.deg2rad(5), 'aoa_el': np.deg2rad(5), 'rss': 3}
C = 3e8  # m/s
FREQ = 915e6  # Hz
REF_RSS = -40  # dBm at 1m
PATH_LOSS_EXP = 2.0
MULTIPATH_RAYS = 2  # Simple: direct + 1 reflection (expand later)

# 6-antenna hex array (symmetric shell; extend to 3D sphere for SO(8) manifold)
if DIM == 2:
    theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
    antennas = ARRAY_RADIUS * np.column_stack([np.cos(theta), np.sin(theta), np.zeros(6)])
else:  # 3D hex approximation (e.g., octahedron-like for triality)
    antennas = ARRAY_RADIUS * np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]  # Placeholder; true hex in 3D could use icosahedral
    ])

# True target pos (x,y,z)
true_pos = np.array([5.0, 3.0, 2.0])[:DIM]

# Compute metrics with simple multipath (direct + wall reflection at z=0 for 3D)
dists_direct = np.linalg.norm(antennas - true_pos, axis=1)
dists_reflect = np.linalg.norm(antennas - true_pos * np.array([1,1,-1]), axis=1) if DIM == 3 else dists_direct
dists = (dists_direct + dists_reflect) / MULTIPATH_RAYS  # Avg for sim; real: resolve components

toas = dists / C
tdoas = toas[1:] - toas[0]
if DIM == 2:
    aoas_az = np.arctan2(true_pos[1] - antennas[:,1], true_pos[0] - antennas[:,0])
    aoas_el = np.zeros_like(aoas_az)  # Flat
else:
    vecs = true_pos - antennas
    aoas_az = np.arctan2(vecs[:,1], vecs[:,0])
    aoas_el = np.arcsin(vecs[:,2] / dists)
rss = REF_RSS - 10 * PATH_LOSS_EXP * np.log10(dists)

# Add noise
tdoas_noisy = tdoas + np.random.normal(0, NOISE_STD['tdoa'], len(tdoas))
aoas_az_noisy = aoas_az + np.random.normal(0, NOISE_STD['aoa_az'], len(aoas_az))
aoas_el_noisy = aoas_el + np.random.normal(0, NOISE_STD['aoa_el'], len(aoas_el))
rss_noisy = rss + np.random.normal(0, NOISE_STD['rss'], len(rss))

# Trisymmetric cost func with normalization
def cost_func(pos_est):
    pos_est = np.pad(pos_est, (0, 3 - DIM))  # Pad for 3D compat
    vecs_est = pos_est - antennas
    dists_est = np.linalg.norm(vecs_est, axis=1)
    toas_est = dists_est / C
    tdoas_est = toas_est[1:] - toas_est[0]
    
    aoas_az_est = np.arctan2(vecs_est[:,1], vecs_est[:,0])
    aoas_el_est = np.arcsin(vecs_est[:,2] / dists_est)
    
    # SO(3) hook: Apply rotations for symmetry (e.g., calibrate array orientation)
    # rot = R.from_euler('z', np.deg2rad(10))  # Example; integrate triality later
    # aoas_az_est = rot.apply(np.column_stack([aoas_az_est, aoas_el_est]))[:,0]  # Stub
    
    rss_est = REF_RSS - 10 * PATH_LOSS_EXP * np.log10(dists_est)
    
    # Errors normalized by variance
    err_tdoa = np.sum((tdoas_noisy - tdoas_est)**2) / (NOISE_STD['tdoa']**2 * len(tdoas))
    err_aoa_az = np.sum((aoas_az_noisy - aoas_az_est)**2) / (NOISE_STD['aoa_az']**2 * len(aoas_az))
    err_aoa_el = np.sum((aoas_el_noisy - aoas_el_est)**2) / (NOISE_STD['aoa_el']**2 * len(aoas_el))
    err_rss = np.sum((rss_noisy - rss_est)**2) / (NOISE_STD['rss']**2 * len(rss))
    
    # Trisymmetric: Equal weight post-norm; SO(8) triality could symmetrize further (e.g., octonion norms)
    return (err_tdoa + err_aoa_az + err_aoa_el + err_rss) / 4

# Optimize
initial_guess = np.zeros(DIM)
result = minimize(cost_func, initial_guess, method='Nelder-Mead')
est_pos = result.x

print(f"True Position: {true_pos}")
print(f"Estimated Position: {est_pos}")
print(f"Error: {np.linalg.norm(est_pos - true_pos)} meters")

# TODO: Integrate SO(8) triality - e.g., represent spectral manifold as 8D vector, use triality automorphisms for one-step fusion
# from octonions import Octonion  # Hypothetical; custom impl for symmetry-preserving optimizer
