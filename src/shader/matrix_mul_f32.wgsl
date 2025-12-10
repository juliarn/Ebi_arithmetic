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

const TILE: u32 = 16;

var<workgroup> Asub : array<array<f32, TILE>, TILE>;
var<workgroup> Bsub : array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE, 1)
fn mul(@builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    let M = dims.n;
    let N = dims.m;
    let K = dims.p;

    var acc: f32 = 0.0;

    let numTiles : u32 = (K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aRow = row;
        let aCol = t * TILE + local_col;
        let bRow = t * TILE + local_row;
        let bCol = col;

        var aVal: f32 = 0.0;
        if (aRow < M && aCol < K) {
            let idxA = aRow * K + aCol;
            aVal = A[idxA];
        }
        Asub[local_row][local_col] = aVal;

        var bVal: f32 = 0.0;
        if (bRow < K && bCol < N) {
            let idxB = bRow * N + bCol;
            bVal = B[idxB];
        }
        Bsub[local_row][local_col] = bVal;

        workgroupBarrier();
        for (var kInner: u32 = 0u; kInner < TILE; kInner = kInner + 1u) {
            acc = acc + Asub[local_row][kInner] * Bsub[kInner][local_col];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        let idxC = row * N + col;
        C[idxC] = acc;
    }
}