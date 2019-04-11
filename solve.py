import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = 'siftflow-fcn32s-heavy.caffemodel'
deploy_proto ='deploy32.prototxt'

# init
caffe.set_mode_cpu() ###

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)
fcn_net = caffe.Net(deploy_proto, weights, caffe.TRAIN) 
surgery.transplant(solver.net,fcn_net) 
del fcn_net


# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('E:/FCN32/sift-flow32s/sift-flow/test.txt', dtype=str)

for _ in range(50):
    solver.step(2000)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    score.seg_tests(solver, False, test, layer='score_sem', gt='sem')
    score.seg_tests(solver, False, test, layer='score_geo', gt='geo')
