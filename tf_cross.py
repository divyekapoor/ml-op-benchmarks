# Author: dkapoor@pinterest.com | Date: Oct 29, 2019
#
# TF 2.0 required. Simply run the code under python 3.
#
# Real world implementation of a cross-feature using Autograph ops.
import time

import tensorflow as tf

# Scroll down till you see the top level function:
# The functions below define the basic UFR style feature inputs.
# The code is the implementation of the following cross feature from C++ code:
# http://opengrok.pinadmin.com/xref/cosmos/scorpion/ads_count/preprocessor_utils.cpp?r=1cbe156e#populateRelatedPinsAnnotationMatchScores
# Outputs

# Enum: ViewType::RELATED_PINS
def sqf_impression_view_type():
    return tf.constant([42])


# map<pterm.PTermType, list<search_common.AnnotationData>>
# The access is only to the ID field inside AnnotationData.
# The 500x is annotation ids. The 100x are PTermTypes.
def sqf_query_annotations():
    return tf.sparse.SparseTensor(
        indices=[[1001, 0], [1002, 0], [1003, 0]],
        values=[5001., 5002., 5003.],
        dense_shape=[10000, 2])  # 1D ID array


# map<pterm.PTermType, list<search_common.AnnotationData>>
# The access is only to the ID field inside AnnotationData.
# The 500x is annotation ids. The 100x are PTermTypes.
def ppd_pin_annotations():
    return tf.sparse.SparseTensor(
        indices=[[1001, 0], [1002, 0], [1003, 0]],
        values=[5001., 5003., 5005.],
        dense_shape=[10000, 2])  # 1D ID array.


# list<PTermType> - constant lookup list for PTerms
def pterm_types():
    return tf.constant([1001, 1002, 1004])


# int32
def task_type():
    return tf.constant(1)


# bool
def need_cmp():
    return tf.constant(False)


# Slice map<PTermType, AnnotationIds> on a constant PTermList used for indexing
# https://github.com/tensorflow/tensorflow/issues/1950
def gather_op(tensor, index_list):
    return tensor
    # return tf.gather(
    #    tensor,
    #    index_list,
    #    validate_indices=True,
    #    axis=0,
    #    name="slice_pterms")


# Implements the cosine max norm match.
# Returns a 2 element list:
# (numMatchesAboveThreshold, cosineMaxNormMatch)
# Tensor1 has shape [10000, 2]
# PyTorch API is better but under-documented:
# https://pytorch.org/docs/stable/sparse.html
def cosine_op(tensor1, tensor2):
    eps_tensor_1 = tf.sparse.SparseTensor(
        tensor1.indices,
        [1e-14] * len(tensor1.values),
        tensor1.dense_shape,
        dtype=tf.float32)
    eps_tensor_2 = tf.sparse.SparseTensor(
        tensor2.indices,
        [1e-14] * len(tensor2.values),
        tensor2.dense_shape,
        dtype=tf.float32)
    minimum_1 = tf.sparse.minimum(eps_tensor_1, tensor1, name='minimum_1')
    minimum_2 = tf.sparse.minimum(eps_tensor_2, tensor2, name='minimum_2')

    max_score_1 = tf.sparse.reduce_max(tensor1, name='maxScore1')

    if max_score_1 < 1e-14:
        return tf.constant([0., 0.])

    if (not tf.reduce_all(tf.math.equal(minimum_1.values, eps_tensor_1.values)) or
            not tf.reduce_all(tf.math.equal(minimum_2.values, eps_tensor_2.values))):
        # Early return
        return tf.constant([0., 0.])

    # Not the most efficient, but ok.
    # tf.sparse_dense_matmul and tf.embedding_lookup_sparse
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/sparse/sparse_dense_matmul
    # Best option is to do a sparse gather + mult but that's taking too much time.
    # https://github.com/tensorflow/tensorflow/issues/1950
    # We can implement this manually, will take time.

    ## Edit: manual implementation
    i = 0
    j = 0
    score = 0.
    i_max = tf.constant(len(tensor1.indices))
    j_max = tf.constant(len(tensor2.indices))
    while i < i_max:
        while j < j_max:
            if tf.reduce_any(tf.math.greater(tensor1.indices[i], tensor2.indices[j])):
                j += 1
            elif tf.reduce_all(tf.math.equal(tensor2.indices[j], tensor1.indices[i])):
                score += tensor1.values[i] * tensor2.values[j]
                j += 1
            else:
                break
        i += 1

    ## Edit: old code
    matmul = tf.matmul(tf.sparse.to_dense(tensor1, 0.0), tf.transpose(tf.sparse.to_dense(tensor2, 0.0)))
    score = tf.reduce_sum(matmul)
    max_score_2 = tf.reduce_max(matmul)
    num_match = tf.math.count_nonzero(matmul)

    divide_1 = tf.math.divide(score, max_score_1, name='div1')
    divide_2 = tf.math.divide(divide_1, max_score_2, name='div2')
    return tf.stack([tf.cast(num_match, tf.dtypes.float32, name='float_cast'), divide_2], axis=0)

###################################################
# Top level function
# Needs to return a shape [10000, 2]
@tf.function
def cross_feature_model(
        sqf_impression_view_type,  # Shape: [1] - int
        sqf_query_annotations,  # Shape: [10000, 2] - float
        ppd_pin_annotations,  # Shape: [10000, 2] - float
        needCmp  # Shape: [1] - bool
):  # Returns cross feature
    if not needCmp or not tf.equal(sqf_impression_view_type, tf.constant(42)):
        return tf.zeros(
            [10000, 2],
            dtype=tf.dtypes.float32,
            name='early_return_zeros')

    # Slice to only the restricted PTermTypes
    pterm_list = pterm_types()
    query_annotations = gather_op(sqf_query_annotations, pterm_list)
    pin_annotations = gather_op(ppd_pin_annotations, pterm_list)

    cosine_max_norm_match = cosine_op(
        query_annotations,
        pin_annotations)

    return cosine_max_norm_match


# Map<PTermType, AnnotationMatchScore>
# PTermType == annotation id
# AnnotationMatchScore == (numMatchesAboveThreshold: i16, cosineMaxNormMatch: double)
def related_pins_annotation_match_score():
    indices = [[1001, 0], [1002, 0], [1003, 0],
               [1001, 1], [1002, 1], [1003, 1]]  # Map entries
    values = [32, 64, 128,  # numMatchesAboveThreshold
              0.1, 0.2, 0.3]  # cosineMaxNorm
    dense_shape = [10000, 2]  # Max number of annotations.
    return tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape)


sqf = {
    'impression.view_type': sqf_impression_view_type()
}

ppd = {

}

intf = {
    'relatedPinsAnnotationMatchScore': related_pins_annotation_match_score()
}

print('Hello world!')

start = time.perf_counter_ns()
mod_result = cross_feature_model(
    sqf_impression_view_type(),
    sqf_query_annotations(),
    ppd_pin_annotations(),
    tf.constant(True))
end = time.perf_counter_ns()
print("Time (usec): ", (end - start) / 1e3)
print('Cosine max norm match: ',
      mod_result)


class Test(tf.Module):

    @tf.function
    def model(self, sqf_impression_view_type,  # Shape: [1] - int
        sqf_query_annotations,  # Shape: [10000, 2] - float
        ppd_pin_annotations,  # Shape: [10000, 2] - float
        needCmp  # Shape: [1] - bool
):  # Returns cross feature
        cross_feature_model(sqf_impression_view_type, sqf_query_annotations, ppd_pin_annotations, needCmp)

mod = Test()
tf.saved_model.save(mod, '/tmp/model.pb')
