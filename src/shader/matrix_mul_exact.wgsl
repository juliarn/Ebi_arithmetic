struct Rational {
    num: u32,
    den: u32,
    sign: u32,
};

struct Dimensions {
    n: u32,
    m: u32,
    p: u32,
};

@group(0) @binding(0)
var<storage, read> A: array<Rational>;

@group(0) @binding(1)
var<storage, read> B: array<Rational>;

@group(0) @binding(2)
var<uniform> dims: Dimensions;

@group(0) @binding(3)
var<storage, read_write> C: array<Rational>;

fn gcd(a: u32, b: u32) -> u32 {
    if (a == u32(0u) || b == u32(0u)) {
        return (a | b);
    }

    let shift = u32(countTrailingZeros(a | b));

    var m = u32(a) >> u32(countTrailingZeros(a));
    var n = u32(b) >> u32(countTrailingZeros(b));

    while (m != n) {
        if (m > n) {
            m -= n;
            m >>= u32(countTrailingZeros(m));
        } else {
            n -= m;
            n >>= u32(countTrailingZeros(n));
        }
    }

    return u32(m << shift);
}

fn lcm(m: u32, n: u32) -> u32 {
    if (m == u32(0u) && n == u32(0u)) {
        return u32(0u);
    }
    return m * (n / gcd(m, n));
}

fn mul_fraction(a: Rational, b: Rational) -> Rational {
    let gdc_ab = gcd(a.num, b.den);
    let gdc_ba = gcd(a.den, b.num);
    let num = (a.num / gdc_ab) * (b.num / gdc_ba);
    let den = (a.den / gdc_ba) * (b.den / gdc_ab);

    var sign: u32 = 1u;
    if (a.sign != b.sign) {
        sign = 0u;
    }
    return Rational(num, den, sign);
}

fn add_fraction(a: Rational,b: Rational) -> Rational {
    if (a.den == b.den) {
        return Rational(a.num + b.num, a.den, 1u);
    }
    let den = lcm(a.den, b.den);
    // Calculate the numerator for a and adjust according to the sign
    var num_a_signed: i64 = i64(a.num * (den / a.den));
    if (a.sign == 0u) {
        num_a_signed = -num_a_signed;
    }
    // Calculate the numerator for b and adjust according to the sign
    var num_b_signed: i64 = i64(b.num * (den / b.den));
    if (b.sign == 0u) {
        num_b_signed = -num_b_signed;
    }
    // Sum the signed numerators
    let num_signed: i64 = num_a_signed + num_b_signed;
    // Determine sign based on sum result
    var sign: u32 = 1u;
    if (num_signed < 0) {
        sign = 0u;
    }
    return Rational(u32(abs(num_signed)), den, sign);
}

@compute @workgroup_size(16, 16)
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

    var rational = Rational(0u, 1u, 1u);
    for (var k = 0u; k < m; k++) {
        let mul = mul_fraction(A[row * m + k], B[k * p + col]);
        if (k == 0u) {
            rational = mul;
        } else {
            rational = add_fraction(rational, mul);
        }
    }

    // Store the result in c
    C[row * p + col] = rational;
}