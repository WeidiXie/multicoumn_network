from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

# Internal
import time
import utils as ut
from multicolumn_network import multiview_network_predict_step1, multiview_network_predict_step2


ut.init_gpu(id='0')
# ------------------------------------------------------------
# load meta information for template-to-template verification.
# tid --> template id,  label --> 1/0
# format:
#           tid_1 tid_2 label
# ------------------------------------------------------------
print('==> start processing meta.....')
meta = np.loadtxt(os.path.join('../meta', 'ijbb_template_pair_label.txt'), dtype=str)
Y = np.array([int(y[-1]) for y in meta])
p1 = np.array([int(p[0]) for p in meta])
p2 = np.array([int(p[1]) for p in meta])
score = np.zeros((len(p1),))   # cls prediction


# ------------------------------------------------------------
# load meta information for extracting image features
# ------------------------------------------------------------
datapath = '../ijbb_crop/'
faces, templates, medias = ut.get_ijbb_datalist(os.path.join('../meta', 'ijbb_face_tid_mid.txt'))
imglist = np.array([os.path.join(datapath, f) for f in faces])
uq_temp = np.unique(templates)
num_uq_temp = len(uq_temp)

# ------------------------------------------------------------
# initialize dictionary for storing features
# template_v : the visual quality only
# template_vc: the visual+content quality
# ------------------------------------------------------------
feature_dict, score_dict = {}, {}
template_v, template_vc = {}, {}


def prediction_step1(networks, batchsize=64):
    '''
        start predicting the features and scores for each image in the dataset
    '''
    sublists = [imglist[i:i + batchsize] for i in range(0, len(imglist), batchsize)]
    sub_length = len(sublists)
    
    for c, s in enumerate(sublists):
        print('==> start processing sublist {}/{}'.format(c, sub_length))
        bz = len(s)
        batch_img = np.zeros((bz, 224, 224, 3))

        # load batches
        for cc, k in enumerate(s):
            batch_img[cc] = ut.load_data_short_axis(k, shape=(224, 224, 3))
        
        # predict batch
        features, scores = networks.predict(batch_img, batch_size=bz)
        
        for cc, k in enumerate(s):
            imgname = k.split(os.sep)[-1]
            feature_dict[imgname] = features[cc]
            score_dict[imgname] = scores[cc][0]


def prediction_step2_template_encoding(networks):
    '''
        compute template encoding,
        take image features, compute media features, and compute template features
    '''

    print('==> start computing template features.')
    img_scores = np.ones((len(faces),))
    img_feats = np.ones((len(faces), 2048))
    for c, img in enumerate(faces):
        img_feats[c] = feature_dict[img]
        img_scores[c] = score_dict[img]
    img_norm_feats = img_feats / np.sqrt(np.sum(img_feats ** 2, -1, keepdims=True))

    for c, uqt in enumerate(uq_temp):
        (ind_t,) = np.where(templates == uqt)
        face_feats = img_feats[ind_t]
        face_norm_feats = img_norm_feats[ind_t]
        face_scores = img_scores[ind_t]
        faces_media = medias[ind_t]
        uqm, counts = np.unique(faces_media, return_counts=True)

        media_feats = []
        media_norm_feats = []
        media_scores = []

        for u, ct in zip(uqm, counts):
            (ind_m, ) = np.where(faces_media == u)
            if ct < 2:
                media_feats += [face_feats[ind_m]]
                media_scores += [face_scores[ind_m[0]]]
                media_norm_feats += [face_norm_feats[ind_m]]
            else:
                media_weighted_feats = face_feats[ind_m] * np.expand_dims(face_scores[ind_m], -1)
                media_feats += [np.sum(media_weighted_feats, 0, keepdims=True) / np.sum(face_scores[ind_m])]
                media_scores += [np.mean(face_scores[ind_m])]
                media_weighted_norm_feats = face_norm_feats[ind_m] * np.expand_dims(face_scores[ind_m], -1)
                media_norm_feats += [np.sum(media_weighted_norm_feats, 0, keepdims=True) / np.sum(face_scores[ind_m])]

        media_scores = np.array(media_scores)

        if len(media_scores.shape) == 2:
            media_scores = np.expand_dims(media_scores,-1)
        elif len(media_scores.shape) == 1:
            media_scores = np.expand_dims(np.expand_dims(media_scores,-1),-1)
        media_feats = np.array(media_feats)

        media_norm_feats = np.array(media_norm_feats)
        media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))

        template_weighted_feats = media_norm_feats * media_scores
        template_norm_feats = np.sum(template_weighted_feats, 0) / np.sum(media_scores)

        # compute the feature weighted by visual quality
        template_v[uqt] = template_norm_feats

        #
        template_feats = media_feats * media_scores
        template_feats = np.sum(template_feats, 0) / np.sum(media_scores)

        arrays = [template_feats for _ in range(media_scores.shape[0])]
        temporary = np.stack(arrays, axis=0)
        
        template_feat_concat = np.concatenate((temporary, media_feats), -1)
        media_re_score = networks.predict(template_feat_concat)

        face_reweighted_feats = media_norm_feats * media_scores * media_re_score
        reweighted_template_norm_feat = np.sum(face_reweighted_feats, 0) / np.sum(media_scores * media_re_score)

        # compute the feature weighted by both visual and content quality
        template_vc[uqt] = reweighted_template_norm_feat


def verification(re_weighted=True):
    # ==========================================================
    #         Loading the Template-specific Features.
    # ==========================================================
    tmp_feats = np.zeros((len(uq_temp), 2048))

    if re_weighted:
        for c, uqt in enumerate(uq_temp):
            tmp_feats[c] = template_vc[uqt]
        print('==> finish loading {} visual-content templates'.format(c))
    else:
        for c, uqt in enumerate(uq_temp):
            tmp_feats[c] = template_v[uqt]
        print('==> finish loading {} visual-only templates'.format(c))

    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    total_pairs = np.array(range(len(Y)))
    batchsize = 128
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(Y), batchsize)]
    for c, s in enumerate(sublists):
        t1 = p1[s]
        t2 = p2[s]
        ind1 = np.squeeze(np.array([np.where(uq_temp == j) for j in t1]))
        ind2 = np.squeeze(np.array([np.where(uq_temp == j) for j in t2]))

        inp1 = tmp_feats[ind1]
        inp2 = tmp_feats[ind2]

        v1 = inp1 / np.sqrt(np.sum(inp1 ** 2, -1, keepdims=True))
        v2 = inp2 / np.sqrt(np.sum(inp2 ** 2, -1, keepdims=True))

        similarity_score = np.sum(v1 * v2, -1)
        score[s] = similarity_score

    if re_weighted:
        print('==> ROC for visual-content multicolumn')
        ut.compute_ROC(Y, score)
    else:
        print('==> ROC for visual-only multicolumn')
        ut.compute_ROC(Y, score)
    return 1


if __name__ == '__main__':
    # ==================================================
    #     Initialize and Pass the Models to Predict
    # ==================================================
    imgdims = (224, 224, 3)
    networks_step1 = multiview_network_predict_step1(input_dim=imgdims)
    networks_step1.load_weights('../models/multicolumn_bmvc.h5', by_name=True)
    print('='*50)
    print('==> loading the networks successfully.')
    print('='*50)
    prediction_step1(networks_step1)

    networks_step2 = multiview_network_predict_step2(input_dim=(1, 4096))
    networks_step2.load_weights('../models/multicolumn_bmvc.h5', by_name=True)
    print('='*50)
    print('==> loading the networks successfully.')
    print('='*50)
    prediction_step2_template_encoding(networks_step2)
    verification(re_weighted=False)
    verification(re_weighted=True)


