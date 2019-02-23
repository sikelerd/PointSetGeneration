import zlib
import numpy as np
from plyfile import PlyElement, PlyData
import cv2

FETCH_BATCH_SIZE = 32
BATCH_SIZE = 32
HEIGHT = 192
WIDTH = 256
POINTCLOUDSIZE = 16384
OUTPUTPOINTS = 1024
REEBSIZE = 1024

binfile = zlib.decompress(open('0/0.gz', 'rb').read())
p = 0
color = np.fromstring(binfile[p:p + FETCH_BATCH_SIZE * HEIGHT * WIDTH * 3], dtype='uint8').reshape((FETCH_BATCH_SIZE, HEIGHT, WIDTH, 3))
color = color[0]
cv2.imwrite('data.png', color)

p += FETCH_BATCH_SIZE * HEIGHT * WIDTH * 3
depth = np.fromstring(binfile[p:p + FETCH_BATCH_SIZE * HEIGHT * WIDTH * 2], dtype='uint16').reshape((FETCH_BATCH_SIZE, HEIGHT, WIDTH))
depth = depth[0]
cv2.imwrite('data_depth.png', depth)

p += FETCH_BATCH_SIZE * HEIGHT * WIDTH * 2
rotmat = np.fromstring(binfile[p:p + FETCH_BATCH_SIZE * 3 * 3 * 4], dtype='float32').reshape((FETCH_BATCH_SIZE, 3, 3))

p += FETCH_BATCH_SIZE * 3 * 3 * 4
ptcloud = np.fromstring(binfile[p:p + FETCH_BATCH_SIZE * POINTCLOUDSIZE * 3], dtype='uint8').reshape((FETCH_BATCH_SIZE, POINTCLOUDSIZE, 3))
ptcloud = ptcloud[0]
ptcloud = ptcloud.astype('float32') / 255
points = list(tuple(ptcloud[i]) for i in range(ptcloud.shape[0]))
points = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
el = PlyElement.describe(points, 'vertex')
plydata = PlyData([el], text=True)
plydata.write('data.ply')
