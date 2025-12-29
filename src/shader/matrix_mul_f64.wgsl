struct Dimensions {
    n: u32,
    m: u32,
    p: u32,
};

@group(0) @binding(0)
var<storage, read> A: array<f64>;

@group(0) @binding(1)
var<storage, read> B: array<f64>;

@group(0) @binding(2)
var<uniform> dims: Dimensions;

@group(0) @binding(3)
var<storage, read_write> C: array<f64>;

const TILING: u32 = 8u;

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

    var sums: array<f64, TILING>;

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
}