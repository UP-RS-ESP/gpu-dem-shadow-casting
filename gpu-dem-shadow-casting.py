import numpy as np
from tqdm import trange
from numba import cuda
from math import sqrt


@cuda.jit("float32[:, :], float32[:, :], int64, int64, int64, float64, float64, float64")
def cuda_kern_cast_shadow(s, z, ys, xs, thr, slope, azim, incl):
    """
    This is the CUDA kernel. Result is written into first argument.
    """
    i, j = cuda.grid(2)
    if i < ys and j < xs:
        b = i - slope * j
        aazim = abs(azim)
        klen = 0
        slen = 0
        k = 0
        o = 0
        if 45 <= aazim and aazim <= 135:
            # steep steps, each row one pixel
            if 0 <= azim:
                for y in range(i, ys):
                    x = int((y - b) / slope)
                    if xs <= x:
                        break
                    if x < 0:
                        break
                    dy, dx = y - i, x - j
                    r = sqrt(dx*dx + dy*dy)
                    dz = z[y, x] - z[i, j] - r * incl
                    if dz > 0:
                        if k == o + 1:
                            klen += 1
                        else:
                            klen = 0
                        if klen > slen:
                            slen = klen
                        o = k
                    if slen > thr:
                        break
                    k += 1
            else:
                for y in range(i, 0, -1):
                    x = int((y - b) / slope)
                    if xs <= x:
                        break
                    if x < 0:
                        break
                    dy, dx = y - i, x - j
                    r = sqrt(dx*dx + dy*dy)
                    dz = z[y, x] - z[i, j] - r * incl
                    if dz > 0:
                        if k == o + 1:
                            klen += 1
                        else:
                            klen = 0
                        if klen > slen:
                            slen = klen
                        o = k
                    if slen > thr:
                        break
                    k += 1
        else:
            # shallow steps, each col one pixel
            if abs(azim) <= 90:
                for x in range(j, xs):
                    y = int(slope * x + b)
                    if ys <= y:
                        break
                    if y < 0:
                        break
                    dy, dx = y - i, x - j
                    r = sqrt(dx*dx + dy*dy)
                    dz = z[y, x] - z[i, j] - r * incl
                    if dz > 0:
                        if k == o + 1:
                            klen += 1
                        else:
                            klen = 0
                        if klen > slen:
                            slen = klen
                        o = k
                    if slen > thr:
                        break
                    k += 1
            else:
                for x in range(j, 0, -1):
                    y = int(slope * x + b)
                    if ys <= y:
                        break
                    if y < 0:
                        break
                    dy, dx = y - i, x - j
                    r = sqrt(dx*dx + dy*dy)
                    dz = z[y, x] - z[i, j] - r * incl
                    if dz > 0:
                        if k == o + 1:
                            klen += 1
                        else:
                            klen = 0
                        if klen > slen:
                            slen = klen
                        o = k
                    if slen > thr:
                        break
                    k += 1
        if slen > thr:
            s[i, j] = 1


def cast_shadow(z, azimuth, inclination):
    """
    This is the CUDA kernel wrapper. First argument is the height image (raster).
    Second argument is the direction of the sun in degrees (-180, 180).
    Third argument is the sun height in degrees (0, 90) with 0 being nadir.
    """
    thr = 5 # minimum light blocking thickness
    assert inclination < 90
    incli = np.tan((90 - inclination) * np.pi / 180)
    slope = np.tan(azimuth * np.pi / 180)
    ys, xs = z.shape
    d_z = cuda.to_device(z.astype("float32"))
    d_s = cuda.device_array((ys, xs), np.float32)
    nthreads = (16, 16)
    nblocksy = ys // nthreads[0] + 1
    nblocksx = xs // nthreads[0] + 1
    nblocks = (nblocksy, nblocksx)
    cuda_kern_cast_shadow[nblocks, nthreads](d_s, d_z, ys, xs, thr, slope, np.float64(azimuth), incli)
    s = d_s.copy_to_host()
    return s


def draw_dome(z, xg, yg, x0, y0, r0):
    """
    Some half spheres for a synthetic elevation model.
    """
    dx = xg - x0
    dy = yg - y0
    sl = np.where(dx*dx + dy*dy < r0*r0)
    dx, dy = dx[sl], dy[sl]
    dz = np.sqrt(r0*r0 - dx*dx - dy*dy)
    z[sl] = np.max((z[sl], dz), axis=0)
    return z


if __name__ == "__main__":
    # example usage
    n = 3 * 1080
    xb = np.arange(16/9*n)
    yb = np.arange(n)
    xr = (xb[1:]+xb[:-1]) / 2
    yr = (yb[1:]+yb[:-1]) / 2
    xg, yg = np.meshgrid(xr, yr)
    z = np.zeros(xg.shape)
    for i in trange(300):
        x0 = np.random.random() * 16 / 9 * n
        y0 = np.random.random() * n
        r0 = np.random.random() * n / 10 + 1
        draw_dome(z, xg, yg, x0, y0, r0)

    height_image = z

    azimuth = 120
    inclination = 60
    shadow_image = cast_shadow(height_image, azimuth, inclination)
    np.save("height_image.npy", height_image)
    np.save("shadow_image.npy", shadow_image)
