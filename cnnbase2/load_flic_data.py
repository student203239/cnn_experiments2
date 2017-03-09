import os

import scipy.io


class FlicLoader(object):

    def __init__(self):
        self.flic_root = "C:\\Users\\Jacek\\Downloads\\flic\\FLIC"
        self.matlab = scipy.io.loadmat(os.path.join(self.flic_root, 'examples.mat'))

    def main(self):
        matlab = self.matlab
        l = len(matlab['examples'][0])
        print l
        print '\n'*3
        print matlab['examples'][0]['torsobox'][99][0]
        print self.torsobox(99)

    def mat_len(self):
        return len(self.matlab['examples'][0])

    def filepath(self, i):
        return self.matlab['examples'][0]['filepath'][i][0]

    def istrain(self, i):
        return self.matlab['examples'][0]['istrain'][i][0][0]

    def istest(self, i):
        return self.matlab['examples'][0]['istest'][i][0][0]

    def torsobox(self, i):
        t = self.matlab['examples'][0]['torsobox'][i][0]
        return t[0], t[1], t[2], t[3]

if __name__ == '__main__':
    loader = FlicLoader()
    loader.main()