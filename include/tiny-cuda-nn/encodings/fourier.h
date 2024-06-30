/** @file   fourier.h
 *  @brief  Implementation of the random Fourier encoding.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>

#include <curand_kernel.h>
#include <random>
#include <vector>

namespace tcnn {

template <typename T>
__global__ void fourier_encoding(
    const uint32_t num_elements,
    const uint32_t input_dims,
    const uint32_t enc_dim,
    const float* __restrict__ B,
    MatrixView<const float> data_in,
    MatrixView<T> data_out
) {
    const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (encoded_index >= num_elements) return;

    const uint32_t i = encoded_index / (enc_dim * 2);
    const uint32_t j = encoded_index % (enc_dim * 2);
    const uint32_t k = j / 2;
    
    float projection = 0.0f;
    for (uint32_t l = 0; l < input_dims; ++l) {
        projection += data_in(l, i) * B[l * enc_dim + k];
    }
    projection *= 2.0f * PI;

    if (j % 2 == 0) {
        data_out(j, i) = __sinf(projection);
    } else {
        data_out(j, i) = __cosf(projection);
    }
}

template <typename T>
__global__ void fourier_encoding_backward(
    const uint32_t num_elements,
    const uint32_t input_dims,
    const uint32_t enc_dim,
    const float* __restrict__ B,
    MatrixView<const T> dL_dy,
    MatrixView<const float> data_in,
    MatrixView<float> dL_dx
) {
    const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (encoded_index >= num_elements) return;

    const uint32_t i = encoded_index / input_dims;
    const uint32_t j = encoded_index % input_dims;

    float gradient = 0.0f;
    for (uint32_t k = 0; k < enc_dim; ++k) {
        float projection = 0.0f;
        for (uint32_t l = 0; l < input_dims; ++l) {
            projection += data_in(l, i) * B[l * enc_dim + k];
        }
        projection *= 2.0f * PI;

        float sin_proj = __sinf(projection);
        float cos_proj = __cosf(projection);

        gradient += dL_dy(2 * k, i) * cos_proj * B[j * enc_dim + k];
        gradient -= dL_dy(2 * k + 1, i) * sin_proj * B[j * enc_dim + k];
    }
    dL_dx(j, i) = gradient * 2.0f * PI;
}

template <typename T>
class FourierEncoding : public Encoding<T> {
public:
    FourierEncoding(uint32_t input_dims, uint32_t enc_dim, float sigma)
        : m_input_dims{input_dims}, m_enc_dim{enc_dim}, m_sigma{sigma} {
        m_output_dims = m_enc_dim * 2;

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0f, m_sigma);

        m_B.resize(m_input_dims * m_enc_dim);
        for (auto& val : m_B) {
            val = distribution(generator);
        }

        cudaMalloc(&d_B, m_B.size() * sizeof(float));
        cudaMemcpy(d_B, m_B.data(), m_B.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~FourierEncoding() {
        cudaFree(d_B);
    }

    std::unique_ptr<Context> forward_impl(
        cudaStream_t stream,
        const GPUMatrixDynamic<float>& input,
        GPUMatrixDynamic<T>* output = nullptr,
        bool use_inference_params = false,
        bool prepare_input_gradients = false
    ) override {
        auto forward = std::make_unique<ForwardContext>();

        if (!output || padded_output_width() == 0) {
            return forward;
        }

        linear_kernel(fourier_encoding<T>, 0, stream,
            input.n() * m_output_dims,
            m_input_dims,
            m_enc_dim,
            d_B,
            input.view(),
            output->view()
        );

        return forward;
    }

    void backward_impl(
        cudaStream_t stream,
        const Context& ctx,
        const GPUMatrixDynamic<float>& input,
        const GPUMatrixDynamic<T>& output,
        const GPUMatrixDynamic<T>& dL_doutput,
        GPUMatrixDynamic<float>* dL_dinput = nullptr,
        bool use_inference_params = false,
        GradientMode param_gradients_mode = GradientMode::Overwrite
    ) override {
        if (!dL_dinput) {
            return;
        }

        linear_kernel(fourier_encoding_backward<T>, 0, stream,
            input.n() * m_input_dims,
            m_input_dims,
            m_enc_dim,
            d_B,
            dL_doutput.view(),
            input.view(),
            dL_dinput->view()
        );
    }

    uint32_t input_width() const override {
        return m_input_dims;
    }

    uint32_t padded_output_width() const override {
        return m_output_dims;
    }

    uint32_t output_width() const override {
        return padded_output_width();
    }

    uint32_t required_input_alignment() const override {
        return 1;
    }

    void set_padded_output_width(uint32_t padded_output_width) override {
        CHECK_THROW(padded_output_width >= m_output_dims);
    }

    uint32_t required_output_alignment() const override {
        return 1;
    }

    MatrixLayout preferred_output_layout() const override {
        return AoS;
    }

    json hyperparams() const override {
        return {
            {"otype", "Fourier"},
            {"num_frequenies", m_enc_dim},
            {"sigma", m_sigma}
        };
    }

private:
    struct ForwardContext : public Context {
    };

    uint32_t m_input_dims;
    uint32_t m_enc_dim;
    uint32_t m_output_dims;
    float m_sigma = 10.0f;

    std::vector<float> m_B;
    float* d_B;
};

}
