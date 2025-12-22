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

@compute @workgroup_size(16, 16)
fn mul(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let M = dims.n;
    let K = dims.m;
    let N = dims.p;

    let row = global_id.y;
    let col = global_id.x * 8;

    if (row >= M || col >= N) {
        return;
    }

    var sum00: f32 = 0.0;
    var sum01: f32 = 0.0;
    var sum02: f32 = 0.0;
    var sum03: f32 = 0.0;
    var sum04: f32 = 0.0;
    var sum05: f32 = 0.0;
    var sum06: f32 = 0.0;
    var sum07: f32 = 0.0;

    for (var i: u32 = 0u; i < K; i = i + 1u) {
        let a_elem = A[row * K + i];
        let b_idx = i * N + col;
        sum00 = fma(a_elem, B[b_idx], sum00);
        sum01 = fma(a_elem, B[b_idx + 1u], sum01);
        sum02 = fma(a_elem, B[b_idx + 2u], sum02);
        sum03 = fma(a_elem, B[b_idx + 3u], sum03);
        sum04 = fma(a_elem, B[b_idx + 4u], sum04);
        sum05 = fma(a_elem, B[b_idx + 5u], sum05);
        sum06 = fma(a_elem, B[b_idx + 6u], sum06);
        sum07 = fma(a_elem, B[b_idx + 7u], sum07);
    }

    let c_idx = row * N + col;

    C[c_idx] = sum00;
    C[c_idx + 1u] = sum01;
    C[c_idx + 2u] = sum02;
    C[c_idx + 3u] = sum03;
    C[c_idx + 4u] = sum04;
    C[c_idx + 5u] = sum05;
    C[c_idx + 6u] = sum06;
    C[c_idx + 7u] = sum07;
}