import torch
import triton
import triton.language as tl

def weighted_sum(x, weight):
    #x has n dim shape [..., D] w has 1d shape [D]
    return (weight * x).sum(axis=-1)

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, #input pointers
    output_ptr, #output pointer
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROW_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr):

    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROW_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROW_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE, ), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, 
                      boundary_check=(0, 1), 
                      padding_option="zero")
        weight = tl.load(weight_block_ptr,
                      boundary_check=(0,),
                      padding_options="zero")

        output += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE))

    tl.store(output_block_ptr, output, boundary_check=(0,))

if __name__ == "__main__":
    x = torch.rand(3, 3)
    w = torch.rand(3)
    print(x)
    print(w)
    print(weighted_sum(x, w))
