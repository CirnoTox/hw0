#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <typeinfo>
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


As an illustration of how to access the data,
note that because X represents a row-major,  𝑚×𝑛  matrix,
if we want to access the  (𝑖,𝑗)  element of  𝑋  (the element in the  𝑖 th row and the  𝑗 th column),
we would use the index X[i*n + j]
*/
void matrixDot(const size_t i, const size_t j, const size_t k,
               const float *A, const float *B, float *result)
{
    // x iter A rows
    for (size_t x = 0; x < i; x++)
    {
        // z iter B cols
        for (size_t z = 0; z < k; z++)
        {
            // y iter A cols and B rows
            // mul and sum
            result[x * k + z] = 0.0f;
            for (size_t y = 0; y < j; y++)
            {
                result[x * k + z] += A[x * j + y] * B[y * k + z];
            }
        }
    }
}
void normalizeVector(const size_t nums, float *V)
{
    float sum = 0.0f;
    for (size_t i = 0; i < nums; i++)
    {
        V[i] = std::exp(V[i]);
        sum += V[i];
    }
    for (size_t i = 0; i < nums; i++)
    {
        V[i] = V[i] / sum;
    }
}
void normalizeMatrix(const size_t rows, const size_t cols, float *M)
{
    for (size_t row = 0; row < rows; row++)
    {
        normalizeVector(cols, M + row * cols);
    }
}
void getIy(const size_t leny, const size_t classNums, const unsigned char *y, float *Iy)
{
    for (size_t i = 0; i < leny; i++)
    {
        for (size_t j = 0; j < classNums; j++)
        {
            Iy[i * classNums + j] = j == (size_t)y[i] ? 1 : 0;
        }
    }
}
void transpose(const size_t rows, const size_t cols, const float *M, float *result)
{
    float tmp[rows * cols] = {};
    size_t count = 0;
    for (size_t j = 0; j < cols; j++)
    {
        for (size_t i = 0; i < rows; i++)
        {
            tmp[count] = M[i * cols + j];
            count++;
        }
    }
    for (size_t i = 0; i < rows * cols; i++)
    {
        result[i] = tmp[i];
    }
}
void matrixMinus(const size_t rows, const size_t cols, float *M1, float *M2, float *result)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result[i * cols + j] = M1[i * cols + j] - M2[i * cols + j];
        }
    }
}
void matrixAdd(const size_t rows, const size_t cols, float *M1, float *M2, float *result)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result[i * cols + j] = M1[i * cols + j] + M2[i * cols + j];
        }
    }
}
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
    float batchX[batch * n] = {};
    unsigned char batchy[batch] = {};

    for (size_t i = 0; i < m; i += batch)
    {
        for(size_t j=0;j<batch*n;j++){
            batchX[j]=X[i*n+j];
        }
        for(size_t j=0;j<batch;j++){
            batchy[j]=y[i+j];
        }
        float Z[batch * k] = {};
        matrixDot(batch, n, k, batchX, theta, Z);
        normalizeMatrix(batch, k, Z);
        float Iy[batch * k] = {};
        getIy(batch, k, batchy, Iy);

        float batchXT[batch * n] = {};
        transpose(batch, n, batchX, batchXT);

        float minusResult[batch * k] = {};
        matrixMinus(batch, k, Z, Iy, minusResult);
        
        float dotResult[n * k] = {};
        matrixDot(n, batch, k, batchXT, minusResult, dotResult);
        float alpha = lr / (float)batch;
        for (size_t j = 0; j < n * k; j++)
        {
            dotResult[j] *= alpha;
        }
        matrixMinus(n, k, theta, dotResult, theta);
    }

    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
