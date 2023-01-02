#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

/*
def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    def normalize(cur):
        def calCuri(curi): 
            return np.exp(curi)/np.sum(np.exp(cur))
        return [*map(calCuri,cur)]
    for index in range(0,len(y),batch):
        batchX,batchy=X[index:index+batch],y[index:index+batch]
        Z=np.array([*map(normalize,np.dot(batchX,theta))])
        newy=np.array([np.concatenate((x[0:yy],np.array([1]),x[yy+1:]), axis=0) 
          for (x,yy) in zip(np.zeros(shape=(batchy.shape[0],theta[0].shape[0])),batchy)
         ])
        theta-=(lr/batch)*np.dot(batchX.transpose(),(Z-newy))

def softmax_loss(Z, y):
    return np.average([*map(lambda idx:np.log(np.sum(np.exp(Z[idx])))-Z[idx][y[idx]],[x for x in range(len(y))])])
    
*/

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    
    
    /// BEGIN YOUR CODE
    //theta[0]=9999.999;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
