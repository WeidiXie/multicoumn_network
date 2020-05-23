import os
import cv2
import numpy as np

mean = (91.4953, 103.8827, 131.0912)


# ------------------------------
# initialize GPUs
# ------------------------------
def init_gpu(id='0'):
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)



# ------------------------------
# meta information
# ------------------------------
def get_ijbb_datalist(s):
    f = open(s, 'rb')
    ijbb_meta = f.readlines()
    faceid = []
    tid = []
    mid = []
    for meta in ijbb_meta:
        j = meta.decode('utf-8').split()
        faceid += [j[0]]
        tid += [int(j[1])]
        mid += [int(j[-1])]

    faceid = np.array(faceid)
    template_id = np.array(tid)
    media_id = np.array(mid)
    return faceid, template_id, media_id


def compute_ROC(labels, scores):
    print('==> compute ROC.')
    import sklearn.metrics as skm
    from scipy import interpolate
    fpr, tpr, thresholds = skm.roc_curve(labels, scores)
    fpr_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    f_interp = interpolate.interp1d(fpr, tpr)
    tpr_at_fpr = [f_interp(x) for x in fpr_levels]
    for (far, tar) in zip(fpr_levels, tpr_at_fpr):
        print('TAR @ FAR = {} : {}'.format(far, tar))


# ------------------------------
# load images
# ------------------------------
def load_data_short_axis(path, shape=(224, 224, 3)):
    # preprocessing, reshape the short size to 256, keep aspect ratio, then do center crop.
    short_size = 256.0
    im = np.asarray(cv2.imread(path), 'uint8')
    im_size = shape
    im_shape = im.shape
    short_axis = 1
    if im_shape != (256, 256, 3):
        if im_shape[0] < im_shape[1]:
            short_axis = 0
            ratio = float(short_size)/im_shape[0]
        else:
            short_axis = 1
            ratio = float(short_size)/im_shape[1]
        im = cv2.resize(im,
                        (int(max(256.0, round(im_shape[1]*ratio))),
                         int(max(256.0, round(im_shape[0]*ratio)))),
                        interpolation=cv2.INTER_LINEAR)

    # im.shape : w x h x 3
    if short_axis == 0:
        st = 16  # center crop
        margin = max(0, int((im.shape[1]-im_size[1])/2))
        temp = im[st:st+im_size[0], margin:margin+im_size[1], :]
    else:
        st = 16  # center crop
        margin = max(0, int((im.shape[0] - im_size[0])/2))
        temp = im[margin: margin+im_size[0], st:st+im_size[1], :]

    return temp - mean
