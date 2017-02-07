#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/affinitize_layer.hpp"

namespace caffe {

template<typename Dtype>
void AffinitizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
{
    AffinitizeParameter param = this->layer_param().affinitize_param();
    channel_axis_ = bottom[0]->CanonicalAxisIndex(param.axis());

    // Affinity distance.
    dst_.clear();
    for (int_tp i = 0; i < static_cast<int_tp>(param.dst_size()); ++i)
    {
        dst_.push_back(param.dst(i));
    }
    CHECK_EQ(dst_.size(), 3) << "AFFINITIZE layer only supports 3D affinity.";
}

template<typename Dtype>
void AffinitizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top)
{
    for (int_tp i = 0; i < bottom.size(); ++i)
    {
        vector<int_tp> top_shape(bottom[i]->shape().begin(),
                                 bottom[i]->shape().end());
        top_shape[channel_axis_] = dst_.size();
        top[i]->Reshape(top_shape);
    }
}

template<typename Dtype>
void AffinitizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_set(top[0]->count(), Dtype(0), top_data);

    // dimension
    int_tp sz = bottom[0]->shape(channel_axis_ + 1);
    int_tp sy = bottom[0]->shape(channel_axis_ + 2);
    int_tp sx = bottom[0]->shape(channel_axis_ + 3);

    // affinity distance
    int_tp dz = dst_[0];
    int_tp dy = dst_[1];
    int_tp dx = dst_[2];

    // stride
    int_tp as = sx*sy*sz;
    int_tp zs = sx*sy;
    int_tp ys = sx;
    int_tp xs = 1;

    for (int_tp z = 0; z < sz; ++z)
        for (int_tp y = 0; y < sy; ++y)
            for (int_tp x = 0; x < sx; ++x)
            {
                int_tp idx = zs*z + ys*y + xs*x;
                Dtype id = bottom_data[idx];

                // x-affinity
                if (x >= dx && id > 0)
                    if (id == bottom_data[idx - xs*dx])
                        top_data[as*0 + idx] = Dtype(1);

                // y-affinity
                if (y >= dy && id > 0)
                    if (id == bottom_data[idx - ys*dy])
                        top_data[as*1 + idx] = Dtype(1);

                // z-affinity
                if (z >= dz && id > 0)
                    if (id == bottom_data[idx - zs*dz])
                        top_data[as*2 + idx] = Dtype(1);
            }
}

INSTANTIATE_CLASS(AffinitizeLayer);
REGISTER_LAYER_CLASS(Affinitize);

}  // namespace caffe
