from numpy import *
from loaddata import *
import matplotlib.pyplot as plt
from pdb import set_trace

DEBUG = True

class RBM:
    """
    the number of augmented visible units = the number of hidden units + 1
    the code uses the number of non-augmented visible units as convention
    """

    def __init__(self):
        
        self.n_h = 64     
        self.n_v = None
        self.W = loadtxt('W.txt')        # n_v+1 - by - n_h
        self.bias_v = loadtxt('bias_v.txt')   # vector a in documentation
        self.bias_h = loadtxt('bias_h.txt')   # vector b in documentation
        self.rate = 5e-6
        self.regLambda = 0e-8

    def sigm(self, x):
        return 1./(1 + exp(-x))

    def loadData(self):
        # load visible data for unsupervised training
        self.images = loadMnist(dataset='training')[0][20000:40000]
        # initialize weights and biases
        self.n_v = self.images[0].size
        #self.W = (random.rand((self.n_v+1) * self.n_h) - .5) * .001
        #self.W = self.W.reshape([self.n_v+1, self.n_h])
        #self.bias_v = (random.rand(self.n_v+1) - .5) * 1e-5
        #self.bias_h[:] = -1.

    def BernoulliSampler(self, prob):
        # prob: probability of being 1
        uniRand = random.rand(prob.size).reshape(prob.shape)
        sample = (uniRand < prob) * 1.
        return sample

    def propV2H(self, v):
        # v: row vector
        # propogate up for hidden layer probability and a sample
        prob_h = self.sigm( dot(hstack([v,1]), self.W) + self.bias_h )
        return prob_h, self.BernoulliSampler(prob_h)

    def propH2V(self, h):
        # h: row vector
        # propogate down for visible layer probability and a sample
        prob_v = self.sigm( ravel(dot(self.W, h[:,newaxis])) + self.bias_v )
        return prob_v, self.BernoulliSampler(prob_v)

    def dataModel(self, i):
        # compute the data terms <vh>, <v>, and <h> for the i-th data
        # CD-3
        v0 = ravel(self.images[i])
        ph0, h0 = self.propV2H(v0)
        vh0 = dot( hstack([v0,1])[:,newaxis], ph0[newaxis,:] )

        v1 = self.propH2V(h0)[1]
        ph1, h1 = self.propV2H(v1[:-1])

        v2 = self.propH2V(h1)[1]
        ph2, h2 = self.propV2H(v2[:-1])

        v3 = self.propH2V(h2)[1]
        ph3, h3 = self.propV2H(v3[:-1])
        vh3 = dot( v3[:,newaxis], ph3[newaxis,:] )

        return vh0, hstack([v0,1]), h0, vh3, v3, h3
        #vh0, v0, h0, vh1, v1, h1 = self.dataModel(0)
        #pv2, v2 = self.propH2V(h1)
        #plt.contourf(v1[:-1].reshape([28,28]),cmap='Greys')

    def gradCD(self, i):
        # compute contrastive-divergence stochastic gradient for sample i
        data_vh, data_v, data_h, model_vh, model_v, model_h = self.dataModel(i)
        return data_vh-model_vh, data_v-model_v, data_h-model_h
    
    def freeEnergy(self):
        # compute free energy of 500 training datas
        freeE = 0
        for i in range(100):
            Ei = dot( hstack([ravel(self.images[i]), 1]), self.W ) + self.bias_h
            freeEi = - sum( log( 1 + exp(Ei) ) ) \
                   - dot( self.bias_v, vstack([ ravel(self.images[i])[:,newaxis], 1 ]) )
            freeE += freeEi
        return freeE

    def train(self):
        epoch = 0
        while True:
            print(self.freeEnergy())
            if epoch%20==1:
                print('saving')
                self.saveWBias()
            for i in range(self.images.shape[0]):
                vhg, vg, hg = self.gradCD(i)
                reg = (vhg > 0.) * 1. # regularizer
                reg = (reg - .5) * 2.
                self.W += (vhg * self.rate) #- self.regLambda * reg
                self.bias_v += (vg * self.rate)
                self.bias_h += (hg * self.rate)
            epoch += 1

    def saveWBias(self):
        # save weights and bias
        savetxt('W.txt', self.W)
        savetxt('bias_v.txt', self.bias_v)
        savetxt('bias_h.txt', self.bias_h)

    def visWeight(self):
        for i in range(8):
            for j in range(8):
                plt.subplot(8,8,i*8+j)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.contourf(self.W[:-1,i*8+j].reshape([28,28]),cmap='Greys')
        plt.show()

    def gibbsTransition(self, i):
        # plot the history of Gibbs sampling transition for the i-th image
        v0 = hstack([ ravel(self.images[i]), 1])
        plt.subplot(1,3,1)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.contourf(v0[:-1].reshape([28,28]), cmap='Greys')

        v = v0
        v[15*28:] = zeros(v0[15*28:].size)
        plt.subplot(1,3,2)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.contourf(v[:-1].reshape([28,28]), cmap='Greys')

        for i in range(30):
            h = self.propV2H(v[:-1])[1]
            v = self.propH2V(h)[0]
            v[:15*28] = v0[:15*28]
        plt.subplot(1,3,3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.contourf(v[:-1].reshape([28,28]), cmap='Greys')
        plt.show()


if __name__ == '__main__':
    rbm = RBM()
    rbm.loadData()
    rbm.train()

