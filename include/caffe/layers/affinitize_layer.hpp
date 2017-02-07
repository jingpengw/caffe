#ifndef CAFFE_AFFINITIZE_LAYER_HPP_
#define CAFFE_AFFINITIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Computes a one edge per dimension 3D affinity graph
 * for a given segmentation/label map
 */
template<typename Dtype>
class AffinitizeLayer : public Layer<Dtype> {
 public:
  explicit AffinitizeLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {
  }

  virtual inline const char* type() const { return "Affinitize";  }
  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp ExactNumTopBlobs()    const { return 1; }
  // virtual inline int_tp MinBottomBlobs() const { return 1; }
  // virtual inline int_tp MaxBottomBlobs() const { return 2; }
  // virtual inline int_tp MinTopBlobs() const { return 1; }
  // virtual inline int_tp MaxTopBlos()  const { return 2; }

 protected:
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //                          const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {}

private:
  std::vector<int_tp> dst_;
  int_tp channel_axis_;
};

}  // namespace caffe

#endif  // CAFFE_AFFINITIZE_LAYER_HPP_
