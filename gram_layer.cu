#include <vector>

#include "caffe/layers/gram_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GramLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* down_sample_data = down_sampled_matrix_.mutable_gpu_data();
	for(int i = 0; i < new_num_slices_; i++) {
		caffe_copy(slice_size_, bottom_data + i * down_stride_ * slice_size_, down_sample_data + i * slice_size_);
	}
	//TODO: Averaging

	Dtype* gram_data = gram_matrix_.mutable_gpu_data();
	for(int i = 0; i < batch_num_; i++) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, down_channel_, down_channel_, slice_size_, (Dtype)1., down_sample_data + i * matrix_size_, down_sample_data + i * matrix_size_, (Dtype)0., gram_data + i * gram_size_);
	}

	Dtype* top_data = top[0]->mutable_gpu_data();
	int offset = 0;
	for(int i = 0; i < batch_num_; i++) {
		for(int j = 0; j < down_channel_; j++) {
			caffe_copy(j + 1, gram_data + i * gram_size_ + j * down_channel_, top_data + offset);
			offset += (j + 1);
		}
	}
	CHECK_EQ(offset, top[0]->count());

}

template <typename Dtype>
void GramLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* gram_diff = gram_matrix_.mutable_gpu_diff();

	int offset = 0;
	for(int i = 0; i < batch_num_; i++) {
		for(int j = 0; j < down_channel_; j++) {
			caffe_copy(j + 1, top_diff + offset, gram_diff + i * gram_size_ + j * down_channel_);
			for(int k = 0; k <= j; k++) {
				caffe_gpu_axpby<Dtype>(1, (Dtype)(1 + (k == j)), top_diff + offset + k, (Dtype)0, gram_diff + i * gram_size_ + k * down_channel_ + j);
			}
			offset += (j + 1);
		}
	}
	
	Dtype* down_sample_diff = down_sampled_matrix_.mutable_gpu_diff();
	Dtype* down_sample_data = down_sampled_matrix_.mutable_gpu_data();
	for(int i = 0; i < batch_num_; i++) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, down_channel_, slice_size_, down_channel_, (Dtype)1., gram_diff + i * gram_size_, down_sample_data + i * matrix_size_, (Dtype)0., down_sample_diff + i * matrix_size_);
	}

	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	caffe_set<Dtype>(bottom[0]->count(), 0, bottom_diff);
	for(int i = 0; i < new_num_slices_; i++) {
		caffe_copy(slice_size_, down_sample_diff + i * slice_size_, bottom_diff + i * down_stride_ * slice_size_);
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(GramLayer);


}  // namespace caffe