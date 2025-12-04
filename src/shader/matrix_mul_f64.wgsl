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

    var sum: f64 = 0.0;
    for (var k = 0u; k < m; k++) {
        sum += A[row * m + k] * B[k * p + col];
    }

    C[row * p + col] = sum; // Store the result in c
}