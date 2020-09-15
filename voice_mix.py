import os
import numpy as np
import tensorflow as tf

from numpy import pi
from tqdm import tqdm
from tensorflow_probability import distributions as tfpd
from tensorflow.contrib.tensorboard.plugins import projector



def Vec_Carte2Ang(vec):
    og_dtype = vec.dtype
    single = tf.equal(tf.size(tf.shape(vec)), 1)
    vec = tf.cond(single, lambda: vec[None], lambda: vec)
    shp = tf.shape(vec)
    print(tf.size(shp))
    #tf.Assert(tf.equal(tf.size(shp), 2), data = [shp])

    vec = tf.cast(vec, dtype=tf.float64)
    dtype = tf.float64 
    '''
    def single():
        vec_len = shp[0]

        mat = tf.sequence_mask(tf.range(vec_len), vec_len, dtype)

        div = tf.square(vec)
        norm = tf.sqrt(tf.reduce_sum(div, axis=0))
        div = tf.reshape(div, [vec_len, 1])
        divisor = tf.reshape(tf.sqrt(tf.matmul(mat, div)), [vec_len, ])

        x0, x1, other = tf.split(vec, [1, 1, vec_len-2])
        _, _, angle_div = tf.split(divisor, [1, 1, vec_len-2])
        angles = tf.atan2(other, angle_div)
        theta0 = tf.atan2(x1, x0)
        theta0 = tf.where(tf.equal(tf.sign(x0), -1), theta0 + pi, theta0)

        return tf.concat([theta0, angles, tf.convert_to_tensor([norm])], axis=0)

    '''
    def batched():
        #[B, vdim] = shp.as_list()
        B, vdim = shp[0], shp[1]

        mat = tf.tile(tf.sequence_mask(tf.range(vdim), vdim, dtype=dtype)[None], [B, 1, 1])     # B x D x D

        div = tf.square(vec)
        norm = tf.sqrt(tf.reduce_sum(div, axis=1))[:, None]     # B x _
        div = tf.reshape(div, [B, vdim, 1])     # B x D x 1
        divisor = tf.reshape(tf.sqrt(tf.matmul(mat, div)), [B, vdim, ])

        x0, x1, other = tf.split(vec, [1, 1, vdim-2], axis=1)
        _, _, angle_div = tf.split(divisor, [1, 1, vdim-2], axis=1)
        angles = tf.atan2(other, angle_div)
        theta0s = tf.atan2(x1, x0)
        theta0s += tf.cast(tf.equal(tf.sign(x0), -1), dtype) * pi

        return tf.concat([theta0s, angles, norm], axis=1)

    ret = batched()
    ret = tf.cond(single, lambda: tf.squeeze(ret, 0), lambda: ret)
    return tf.cast(ret, og_dtype)


def Vec_Ang2Carte(ang_vec):
    og_dtype = ang_vec.dtype
    single = tf.equal(tf.size(tf.shape(ang_vec)), 1)
    ang_vec = tf.cond(single, lambda: ang_vec[None], lambda: ang_vec)
    shp = tf.shape(ang_vec)
    print(tf.size(shp))
    #tf.Assert(tf.equal(tf.size(shp), 2), data= [shp])

    ang_vec = tf.cast(ang_vec, dtype=tf.float64)
    dtype = tf.float64
    '''
    def single():
        vec_len = shp[0]

        theta0, angles, Norm = tf.split(ang_vec, [1, vec_len-2, 1])
        over = tf.cast(tf.greater(theta0,(0.5*pi)), dtype)
        tf.add(theta0, -pi*over)
        angles = tf.concat([theta0, angles], axis=0)
        w_c = tf.math.cumprod(tf.concat([tf.cos(angles), Norm], axis=0), reverse=True)

        mult = tf.concat([tf.cast(tf.convert_to_tensor([1]), dtype=dtype), tf.cast(tf.math.sin(angles), dtype=dtype)], axis=0)
        ret = tf.multiply(w_c, mult)
        negator = -2*over*tf.cast(tf.pad([1, 1], [[0, vec_len-2]]), dtype)+tf.ones(vec_len, dtype)
        return tf.multiply(ret, negator)
    '''
    def batched():

        #[B, vdim] = ang_vec.get_shape.as_list()
        B, vdim = shp[0], shp[1]

        theta0s, angles, Norm = tf.split(ang_vec, [1, vdim-2, 1], axis=1)
        over = tf.cast(tf.greater(theta0s, 0.5*pi*tf.ones(tf.shape(theta0s), dtype)), dtype=dtype)
        tf.add(theta0s, -pi*over)
        angles = tf.concat([theta0s, angles], axis=1)
        w_c = tf.math.cumprod(tf.concat([tf.cos(angles), Norm], axis=1), axis=1, reverse=True)

        mult = tf.concat([tf.reshape(tf.cast(tf.convert_to_tensor([1]*B), dtype),[B, 1]), tf.cast(tf.math.sin(angles), dtype)], axis=1)

        ret = tf.multiply(w_c, mult)
        negator = tf.matmul(-2*over, tf.cast(tf.pad([1, 1], [[0, vdim-2]])[None], dtype)) + tf.ones(shape=[B, vdim], dtype=dtype)
        return tf.multiply(ret, negator)

    ret = batched()
    ret = tf.cond(single, lambda: tf.squeeze(ret, 0), lambda: ret)
    return tf.cast(ret, og_dtype)


def Vector_Angular_Mix(vec1, vec2, state1='Cartesian', state2='Cartesian', out_state='Cartesian', mix_ratio=0.5):

    assert 0. <= mix_ratio and mix_ratio <= 1.
    assert vec1.get_shape() == vec2.get_shape()

    if state1 == 'Cartesian':
        vec1 = Vec_Carte2Ang(vec1)
    vec_b = vec2
    if state2 == 'Cartesian':
        vec2 = Vec_Carte2Ang(vec2)

    vector = mix_ratio*vec1 + (1.-mix_ratio)*vec2
    if out_state == 'Cartesian':
        vector = Vec_Ang2Carte(vector)

    return vector



'''
From Now, Multivariate GMM for clusterings
'''

class MVGMM:

    def __init__(self, data, num_cluster, dtype=tf.float64):
        '''
        N : number of data
        D : space dimension
        K : number of clustering means
        gamma : gamma values
        '''
        with tf.variable_scope('Values'):
            self.dtype = dtype
            [self.N, self.D] = data.get_shape().as_list()
            self.K = num_cluster
            self.data = tf.convert_to_tensor(data, dtype=self.dtype) # N x D

            self.log_Gs = tf.Variable(
                    tf.log(tf.ones([self.N, self.K], dtype=self.dtype)/self.K),
                    name='log_Gs')
            self.means = tf.Variable(
                    tf.random.normal([self.K, self.D], mean=tf.reduce_mean(data, axis=0),dtype=self.dtype),
                    name='means')
            self.scales = tf.Variable(
                    tf.linalg.diag(tf.random.uniform(shape=[self.K, self.D], minval=0.9, maxval=1.1, dtype=self.dtype), k=self.K, num_rows=self.D),
                    name='scales')     # K x D x D
            self.groups = tf.Variable(
                    tf.zeros(self.N, dtype=tf.argmax(self.means).dtype),
                    name='groups')


    def get(self, mode='Nk'):
        with tf.variable_scope('get_'+mode):
            if mode == 'Nk':
                ret = tf.reduce_logsumexp(self.log_Gs, axis=0)
            else:
                raise NotImplementedError
        return ret


    def E(self):
        with tf.variable_scope('E_step'):
            # K_Gauss = tfpd.MultivariateNormalFullCovariance(loc=self.means,covariance_matrix=self.scale)
            # K_Gauss = tfpd.MultivariateNormalTriL(loc=self.means, scale_tril=self.scale)

            DminM = (self.data[:, None]-self.means[None])[:, :, :, None]     # N x K x D x 1
            #s, u, v = tf.svd(self.scales)
            #atol = tf.reduce_max(s) * 1e-16
            #s = tf.boolean_mask(s, s>atol)
            #s_inv = tf.diag(1. / s)
            #inversed = tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))
            y = tf.matmul(tf.transpose(DminM, [0, 1, 3, 2]),
                    tf.matmul(tf.tile(tf.linalg.pinv(self.scales)[None], [self.N, 1, 1, 1]), DminM))     # N x K x _
            y = tf.squeeze(y, [2,3])
            log_z = 0.5*tf.math.log(tf.tile(tf.abs(tf.linalg.det(self.scales)[None]), [self.N, 1]))     # N x K x _

            #print('y', y)
            #print('logz', log_z)
            log_pdf = -tf.cast(0.5, self.dtype)*y - log_z     # N x K x _
            # print(log_pdf)
            #scaled_pdf = log_pdf - tf.transpose([tf.reduce_max(log_pdf, axis=1)]*self.K, [1, 0])
            #print('scaled', log_pdf)

            log_Nk = self.get()     # K x _
            log_mat = log_Nk[None] + log_pdf    # N x K
            #print('log Nks in E', log_Nk)

            l = tf.reduce_logsumexp(log_mat, axis=1, keepdims=True)
            log_Gs = log_mat-l

            #print('l', log_P)
            return log_Gs, l


    def M(self):
        with tf.variable_scope('M_step'):
            #print('gamma', self.log_Gs)
            log_Nks = self.get()
            #print('log Nks', log_Nks)

            #means = tf.matmul(mult, tf.matmul(tf.transpose(self.log_Gs), self.data))
            means = tf.reduce_sum(tf.exp(self.log_Gs[:, :, None])*self.data[:, None], axis=0)/tf.exp(log_Nks[:, None])

            #mat = tf.subtract(tf.transpose([self.data]*self.K, [1, 0, 2]),
            #        tf.convert_to_tensor([means]*self.N))[:, :, :, None]
            mat = (self.data[:, None, :] - means[None])[:, :, :, None]     # N x K x D
            matt = tf.matmul(mat, tf.transpose(mat, [0, 1, 3, 2]))    # N x K x D x D
            matt *= tf.exp(self.log_Gs[:, :, None, None])
            #print('numer', tf.reduce_sum(matt, axis=0))
            #print('denom', tf.exp(log_Nks[:, None, None]))
            scale = tf.reduce_sum(matt, axis=0)/tf.exp(log_Nks[:, None, None])     # K x D x D

            # eigen decomposition used
            # zero determinant control
            #print('scale', tf.reduce_min(scale), tf.reduce_max(scale), scale)
            #print('det1', tf.linalg.det(scale))
            es, vs = tf.linalg.eigh(scale)
            diag = es + 1e-1*tf.cast(tf.less(es, 1e-2), dtype=self.dtype)     # K x D

            # negative determinant control for es
            rec_scale = tf.matmul(tf.matmul(vs, tf.linalg.diag(tf.abs(diag))), tf.transpose(vs, [0, 2, 1]))
            #print('det2', tf.linalg.det(rec_scale))

            # too small determinant, then re-initialize the cov
            flag = tf.cast(tf.equal(tf.linalg.det(rec_scale), 0), self.dtype)[:, None, None]
            fixed_scale = tf.add(rec_scale*(tf.cast(1, self.dtype)-flag),
                    flag*tf.linalg.diag(tf.random.uniform(shape=[self.K, self.D], minval=0.9, maxval=1.1, dtype=self.dtype), k=self.K, num_rows=self.D)) # *tf.eye(self.D, batch_shape=[self.K], dtype=self.dtype)     # K x D x D
            #print('fixed_scale', tf.linalg.det(fixed_scale))

        return means, fixed_scale


    def save(self, session, mode='csv', log_dir='logs/data-mix/', name='data_cluster', opt='a', 
            switch={'means':1, 'scales':1, 'log_Gs':0, 'groups':1}):

        with tf.variable_scope('save'):
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            mean_save = self.means.eval(session)     # means: K x D
            scale_save = self.scales.eval(session)     # scale: K x D x D
            gamma_save = self.log_Gs.eval(session)     # log_Gs: N x K
            group = self.groups.eval(session)     # groups: N x _

            if mode=='csv':
                if switch['means']:
                    with open(os.path.join(log_dir, name+'_mean.csv'), opt) as f:
                        for rows in mean_save:
                            for items in rows:
                                f.write("%f," % items)
                            f.write("\n")
                        f.write("\n")
                if switch['scales']:
                    with open(os.path.join(log_dir, name+'_scale.csv'), opt) as f:
                        for kth in scale_save:
                            for rows in kth:
                                for elem in rows:
                                    f.write("%f," % elem)
                                f.write("\n")
                            f.write("\n")
                        f.write("\n")
                if switch['log_Gs']:
                    with open(os.path.join(log_dir, name+'_log_G.csv'), opt) as f:
                        for nth in gamma_save:
                            for elem in nth:
                                f.write("%f," % elem)
                            f.write("\n")
                        f.write("\n")
                if switch['groups']:
                    with open(os.path.join(log_dir, name+'_group.csv'), opt) as f:
                        for kth in group:
                            f.write("%f\n" % kth)
                        f.write("\n")
            elif mode=='npy':
                if switch['means']:
                    np.save(os.path.join(log_dir, name+'_mean.npy'), np.array(mean_save))
                if switch['scales']:
                    np.save(os.path.join(log_dir, name+'_scale.npy'), np.array(scale_save))
                if switch['log_Gs']:
                    np.save(os.path.join(log_dir, name+'_log_G.npy'), np.array(gamma_save))
                if switch['groups']:
                    np.save(os.path.join(log_dir, name+'_group.npy'), np.array(group))
            else:
                raise NotImplementedError


    def train(self, tolerance=1e-300, max_steps=100001,
            ckptdir = 'logs', name='who', wtype='w', ftype='csv', feedback=None):

        with tf.variable_scope('train'):
            log_Gam, l = self.E()
            opE = tf.assign(self.log_Gs, log_Gam)
            oploglike = tf.reduce_sum(l)

            m, c = self.M()
            opM1 = tf.assign(self.means, m)
            opM2 = tf.assign(self.scales, c)

            group = tf.assign(self.groups, tf.argmax(log_Gam, axis=1))
            opgroup = tf.assign(self.groups, group)

            namelen = len(str(max_steps))


            # Start
            config=tf.ConfigProto()
            config.gpu_options.allow_growth = True

            ckpt_dir = os.path.join(ckptdir, 'checkpoints', name, wtype)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                prev_llh = 10

                for step in tqdm(range(max_steps), dynamic_ncols=True):
                    _, curr_llh, _, _ = sess.run([opE, oploglike, opM1, opM2])

                    if np.abs(curr_llh-prev_llh) < tolerance:
                        sess.run(opgroup)
                        self.save(sess, mode=ftype, log_dir=ckpt_dir, 
                                name='0'*(namelen-len(str(step)))+str(step))
                        break

                    prev_llh = curr_llh

                    if step%500 == 0:
                        sess.run(opgroup)
                        self.save(sess, mode=ftype, log_dir=ckpt_dir, 
                                name='0'*(namelen-len(str(step)))+str(step))



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_ = tf.concat([tf.random.normal(shape=[10,2], mean=[5, 10], dtype=tf.float64),
        tf.random.normal(shape=[10,2], mean=[20, 1], dtype=tf.float64),
    tf.random.normal(shape=[10,2], mean=[30, 30], dtype=tf.float64)], axis=0)

    kmeans = MVGMM(data=data_, num_cluster=3, dtype=data_.dtype)
    kmeans.train(name='data_test2', ftype='npy')

    print(np.array(kmeans.means))
    print(np.array(kmeans.log_Gs))


