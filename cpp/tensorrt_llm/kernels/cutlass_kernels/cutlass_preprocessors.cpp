constexpr int BITS_PER_BYTE = 8;
constexpr int BYTES_PER_WORD = 4;
constexpr int BITS_PER_WORD = BITS_PER_BYTE * BYTES_PER_WORD;
constexpr int ELTS_PER_WORD = BITS_PER_WORD / get_bits_in_quant_type(quant_type);


void transpose_subbytes(int8_t* transposed_quantized_tensor, const int8_t* quantized_tensor,
    const std::vector<size_t>& shape, QuantType quant_type)
{
    // ...

    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        const size_t matrix_offset = expert * num_rows * col_bytes;
        for (size_t row_tile_start = 0; row_tile_start < num_rows; row_tile_start += M_TILE_L1)
        {
            for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes;
                 col_tile_start_byte += N_TILE_L1)
            {
                transpose_subbytes_impl<QuantType::INT8_WEIGHT_ONLY>(
                    transposed_quantized_tensor, quantized_tensor, shape, expert, row_tile_start,
                    col_tile_start_byte);
            }
        }
    }
}

template <QuantType quant_type>
void transpose_subbytes_impl(int8_t* transposed_quantized_tensor, const int8_t* quantized_tensor,
    const std::vector<size_t>& shape, size_t expert, size_t row_tile_start, size_t col_tile_start_byte)
{
    // ...
}


class Int4
{
public:
    Int4(int8_t a, int8_t b) : a(a), b(b) {}

    int8_t a;
    int8_t b;
};

Int4 pack_int4(int8_t a, int8_t b)
{
    Int4 int4;
    int4.a = a;
    int4.b = b;
    return int4;
}

int8_t unpack_int4_a(const Int4& int4)
{
    return int4.a;
}

int8_t unpack_int4_b(const Int4& int4)
{
    return int4.b;
}

void interleave_int4s(int8_t* interleaved_quantized_tensor, const Int4* quantized_tensor,
    const std::vector<size_t>& shape, QuantType quant_type)
{
    // ...

    for (int ii = 0; ii < num_elts; ii += 2)
    {
        int8_t a = unpack_int4_a(quantized_tensor[ii]);
        int8_t b = unpack_int4_b(quantized_tensor[ii + 1]);

        interleaved_quantized_tensor[ii / 2] = pack_int4(a, b);
    }
}


const uint32_t* input_byte_ptr = static_cast<const uint32_t*>(quantized_tensor);
uint32_t* output_byte_ptr = static_cast<uint32_t*>(permuted_quantized_tensor);


if (quant_type != QuantType::INT8_WEIGHT_ONLY && quant_type != QuantType::PACKED_INT4_WEIGHT_ONLY)
{
    throw std::invalid_argument("Unsupported quantization type");
}
