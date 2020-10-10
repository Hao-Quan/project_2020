import h5py
import numpy as np

# path = '../ntu_light/NTU_CS.h5'
path = '/data/ntu/NTU_CS.h5'

f = h5py.File(path, 'r')
x = np.concatenate([f['x'][:], f['x'][:]], axis=1)
y = np.argmax(f['y'][:] ,-1)
valid_x = np.concatenate([f['valid_x'][:], f['valid_x'][:]], axis=1)
valid_y = np.argmax(f['valid_y'][:], -1)
test_x = np.concatenate([f['test_x'][:], f['test_x'][:]], axis=1)
test_y = np.argmax(f['test_y'][:], -1)
f.close()

# with open('../ntu_light/h5/xsub/x.npy', 'wb') as fnpy:
#     print("save x.npy")
#     np.save(fnpy, x)
#
# with open('../ntu_light/h5/xsub/y.npy', 'wb') as fnpy:
#     print("save y.npy")
#     np.save(fnpy, y)
#
# with open('../ntu_light/h5/xsub/valid_x.npy', 'wb') as fnpy:
#     print("save valid_x.npy")
#     np.save(fnpy, valid_x)
#
# with open('../ntu_light/h5/xsub/valid_y.npy', 'wb') as fnpy:
#     print("save valid_y.npy")
#     np.save(fnpy, valid_y)
#
# with open('../ntu_light/h5/xsub/test_x.npy', 'wb') as fnpy:
#     print("save test_x.npy")
#     np.save(fnpy, test_x)
#
# with open('../ntu_light/h5/xsub/test_y.npy', 'wb') as fnpy:
#     print("save test_y.npy")
#     np.save(fnpy, test_y)

with open('/data/ntu/h5/xsub/x.npy', 'wb') as fnpy:
    print("save x.npy")
    np.save(fnpy, x)

print("start y.npy")
with open('/data/ntu/h5/xsub/y.npy', 'wb') as fnpy:
    print("save y.npy")
    np.save(fnpy, y)

with open('/data/ntu/h5/xsub/valid_x.npy', 'wb') as fnpy:
    print("save valid_x.npy")
    np.save(fnpy, valid_x)

with open('/data/ntu/h5/xsub/valid_y.npy', 'wb') as fnpy:
    print("save valid_y.npy")
    np.save(fnpy, valid_y)

with open('/data/ntu/h5/xsub/test_x.npy', 'wb') as fnpy:
    print("save test_x.npy")
    np.save(fnpy, test_x)

with open('/data/ntu/h5/xsub/test_y.npy', 'wb') as fnpy:
    print("save test_y.npy")
    np.save(fnpy, test_y)