struct SignedU64 {
    value: u64,
    sign: u64,
}

struct Dimensions {
    n: u32,
    m: u32,
    p: u32,
};

@group(0) @binding(0)
var<storage, read> A: array<SignedU64>;

@group(0) @binding(1)
var<storage, read> B: array<SignedU64>;

@group(0) @binding(2)
var<uniform> dims: Dimensions;

@group(0) @binding(3)
var<storage, read_write> C: array<SignedU64>;

fn multiply_signed_u64(a: SignedU64, b: SignedU64) -> SignedU64 {
    let product_value = a.value * b.value;
    let product_sign = a.sign ^ b.sign; // XOR to determine the sign of the product
    return SignedU64(product_value, product_sign);
}

fn add_signed_u64(a: SignedU64, b: SignedU64) -> SignedU64 {
    if (a.sign == b.sign) {
        // Same sign, add the values
        return SignedU64(a.value + b.value, a.sign);
    } else {
        // Different signs, subtract the smaller from the larger
        if (a.value > b.value) {
            return SignedU64(a.value - b.value, a.sign);
        } else if (b.value > a.value) {
            return SignedU64(b.value - a.value, b.sign);
        } else {
            // They are equal, result is zero
            return SignedU64(u64(0u), u64(1u));
        }
    }
}

fn multiply_add_signed_u64(a: SignedU64, b: SignedU64, acc: SignedU64) -> SignedU64 {
    let product = multiply_signed_u64(a, b);
    return add_signed_u64(acc, product);
}

const TILING: u32 = 8u;

@compute @workgroup_size(16, 16, 1)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = dims.n;
    let m = dims.m;
    let p = dims.p;

    let row = global_id.x;
    let col = global_id.y * TILING;

    if (row >= n || col >= p) {
        return;
    }

    var sums: array<SignedU64, TILING> = array<SignedU64, TILING>(
        SignedU64(u64(0u), u64(1u)),
        SignedU64(u64(0u), u64(1u)),
        SignedU64(u64(0u), u64(1u)),
        SignedU64(u64(0u), u64(1u)),
        SignedU64(u64(0u), u64(1u)),
        SignedU64(u64(0u), u64(1u)),
        SignedU64(u64(0u), u64(1u)),
        SignedU64(u64(0u), u64(1u)),
    );

    for (var k: u32 = 0u; k < m; k++) {
        let a_elem = A[row * m + k];
        let b_idx = k * p + col;

        sums[0] = multiply_add_signed_u64(a_elem, B[b_idx + 0u], sums[0]);
        if (col + 1u < p) {
            sums[1] = multiply_add_signed_u64(a_elem, B[b_idx + 1u], sums[1]);
        }
        if (col + 2u < p) {
            sums[2] = multiply_add_signed_u64(a_elem, B[b_idx + 2u], sums[2]);
        }
        if (col + 3u < p) {
            sums[3] = multiply_add_signed_u64(a_elem, B[b_idx + 3u], sums[3]);
        }
        if (col + 4u < p) {
            sums[4] = multiply_add_signed_u64(a_elem, B[b_idx + 4u], sums[4]);
        }
        if (col + 5u < p) {
            sums[5] = multiply_add_signed_u64(a_elem, B[b_idx + 5u], sums[5]);
        }
        if (col + 6u < p) {
            sums[6] = multiply_add_signed_u64(a_elem, B[b_idx + 6u], sums[6]);
        }
        if (col + 7u < p) {
            sums[7] = multiply_add_signed_u64(a_elem, B[b_idx + 7u], sums[7]);
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