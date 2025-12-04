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

@compute @workgroup_size(16, 16, 1)
fn mul(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = dims.n; // Number of rows of a
    let m = dims.m; // Number of columns of a (and rows of b)
    let p = dims.p; // Number of columns of b

    let row = id.x; // Row index in the result matrix c
    if (row >= n) {
        return; // Out of bounds
    }
    let col = id.y; // Column index in the result matrix c
    if (col >= p) {
        return; // Out of bounds
    }

    var sum: SignedU64 = SignedU64(u64(0u), u64(1u)); // Initialize sum to zero
    for (var k = 0u; k < m; k++) {
        let product = multiply_signed_u64(A[row * m + k], B[k * p + col]);
        sum = add_signed_u64(sum, product);
    }

    C[row * p + col] = sum; // Store the result in c
}