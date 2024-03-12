/**
 * @brief The EpiloguePerRowPerColScale class is a custom epilogue that scales the output matrix
 *        by a row and column scale factor.
 */
class EpiloguePerRowPerColScale : public cutlass::gemm::kernel::Epilogue
{
public:
    using ElementCompute = float;
    using ElementOutput = float;
    using Layout = layout::RowMajor;

    /**
     * @brief Construct a new EpiloguePerRowPerColScale object
     *
     * @param alpha_row The row scale factor
     * @param alpha_col The column scale factor
     */
    EpiloguePerRowPerColScale(ElementCompute alpha_row, ElementCompute alpha_col)
        : alpha_row_(alpha_row), alpha_col_(alpha_col)
    {
    }

    /**
     * @brief The Visitor class is used to visit each element in the output matrix and apply the
     *        row and column scale factors.
     */
    class Visitor
    {
    public:
        using ElementCompute = float;
        using ElementOutput = float;

        /**
         * @brief Construct a new Visitor object
         *
         * @param alpha_row The row scale factor
         * @param alpha_col The column scale factor
         */
        Visitor(ElementCompute alpha_row, ElementCompute alpha_col)
            : alpha_row_(alpha_row), alpha_col_(alpha_col)
        {
        }

        /**
         * @brief Visit an element in the output matrix and apply the row and column scale factors.
         *
         * @param element The element to visit
         * @param thread_idx The thread index
         * @param warp_idx The warp index
         * @param lane_idx The lane index
         * @param params The parameters for the visitor
         */
        CUTLASS_DEVICE
        void operator()(ElementOutput& element, int thread_idx, int warp_idx, int lane_idx,
            typename Epilogue::Params const& params) const
        {
            // Apply the row scale factor
            ElementCompute row_scale = alpha_row_ * params.alpha_row[warp_idx];

            // Apply the column scale factor
            ElementCompute col_scale = alpha_col_ * params.alpha_col[thread_idx];

            // Multiply the element by the row and column scale factors
            element *= row_scale * col_scale;
        }

    private:
        ElementCompute alpha_row_;
        ElementCompute alpha_col_;
    };

    /**
     * @brief Get the visitor object
     *
     * @return The visitor object
     */
    CUTLASS_DEVICE
    Epilogue::Visitor const& get_visitor() const
    {
        return visitor_;
    }

    /**
     * @brief Get the number of elements per access
     *
     * @return The number of elements per access
     */
    CUTLASS_DEVICE
    int get_elements_per_access() const
    {
        return 1;
    }

private:
    ElementCompute alpha_row_;
    ElementCompute alpha_col_;

    Visitor visitor_;
};

