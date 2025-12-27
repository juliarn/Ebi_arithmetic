struct Dimensions {
    n: u32,
    m: u32,
    p: u32,
};

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<uniform> dims: Dimensions;

@group(0) @binding(3)
var<storage, read_write> C: array<f32>;

const TILING: u32 = 16u;

@compute @workgroup_size(16, 16)
fn mul(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let n = dims.n;
    let m = dims.m;
    let p = dims.p;

    let row = global_id.x;
    let col = global_id.y * TILING;

    if (row >= n || col >= p) {
        return;
    }

    var sums: array<f32, TILING>;

    for (var k: u32 = 0u; k < m; k++) {
        let a_elem = A[row * m + k];
        let b_idx = k * p + col;

        sums[0] = fma(a_elem, B[b_idx + 0u], sums[0]);
        if (col + 1u < p) {
            sums[1] = fma(a_elem, B[b_idx + 1u], sums[1]);
        }
        if (col + 2u < p) {
            sums[2] = fma(a_elem, B[b_idx + 2u], sums[2]);
        }
        if (col + 3u < p) {
            sums[3] = fma(a_elem, B[b_idx + 3u], sums[3]);
        }
        if (col + 4u < p) {
            sums[4] = fma(a_elem, B[b_idx + 4u], sums[4]);
        }
        if (col + 5u < p) {
            sums[5] = fma(a_elem, B[b_idx + 5u], sums[5]);
        }
        if (col + 6u < p) {
            sums[6] = fma(a_elem, B[b_idx + 6u], sums[6]);
        }
        if (col + 7u < p) {
            sums[7] = fma(a_elem, B[b_idx + 7u], sums[7]);
        }
        if (col + 8u < p) {
            sums[8] = fma(a_elem, B[b_idx + 8u], sums[8]);
        }
        if (col + 9u < p) {
            sums[9] = fma(a_elem, B[b_idx + 9u], sums[9]);
        }
        if (col + 10u < p) {
            sums[10] = fma(a_elem, B[b_idx + 10u], sums[10]);
        }
        if (col + 11u < p) {
            sums[11] = fma(a_elem, B[b_idx + 11u], sums[11]);
        }
        if (col + 12u < p) {
            sums[12] = fma(a_elem, B[b_idx + 12u], sums[12]);
        }
        if (col + 13u < p) {
            sums[13] = fma(a_elem, B[b_idx + 13u], sums[13]);
        }
        if (col + 14u < p) {
            sums[14] = fma(a_elem, B[b_idx + 14u], sums[14]);
        }
        if (col + 15u < p) {
            sums[15] = fma(a_elem, B[b_idx + 15u], sums[15]);
        }
    }

    let c_idx = row * p + col;

    C[c_idx + 0u] = sums[0];
    if (col + 1u < p) {
        C[c_idx + 1u] = sums[1];
    }
    if (col + 2u < p) {
        C[c_idx + 2u] = sums[2];
    }
    if (col + 3u < p) {
        C[c_idx + 3u] = sums[3];
    }
    if (col + 4u < p) {
        C[c_idx + 4u] = sums[4];
    }
    if (col + 5u < p) {
        C[c_idx + 5u] = sums[5];
    }
    if (col + 6u < p) {
        C[c_idx + 6u] = sums[6];
    }
    if (col + 7u < p) {
        C[c_idx + 7u] = sums[7];
    }
    if (col + 8u < p) {
        C[c_idx + 8u] = sums[8];
    }
    if (col + 9u < p) {
        C[c_idx + 9u] = sums[9];
    }
    if (col + 10u < p) {
        C[c_idx + 10u] = sums[10];
    }
    if (col + 11u < p) {
        C[c_idx + 11u] = sums[11];
    }
    if (col + 12u < p) {
        C[c_idx + 12u] = sums[12];
    }
    if (col + 13u < p) {
        C[c_idx + 13u] = sums[13];
    }
    if (col + 14u < p) {
        C[c_idx + 14u] = sums[14];
    }
    if (col + 15u < p) {
        C[c_idx + 15u] = sums[15];
    }
}