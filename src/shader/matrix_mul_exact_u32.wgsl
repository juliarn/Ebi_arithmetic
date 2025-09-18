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

struct U32Overflow {
    value: u32,
    overflow: bool,
}

const OVERFLOW = Rational(0xffffffffu, 0u, 1u);

@group(0) @binding(0)
var<storage, read> A: array<Rational>;

@group(0) @binding(1)
var<storage, read> B: array<Rational>;

@group(0) @binding(2)
var<uniform> dims: Dimensions;

@group(0) @binding(3)
var<storage, read_write> C: array<Rational>;

fn gcd(a: u32, b: u32) -> u32 {
    if (a == 0u || b == 0u) {
        return (a | b);
    }

    let shift = u32(countTrailingZeros(a | b));

    var m = a >> u32(countTrailingZeros(a));
    var n = b >> u32(countTrailingZeros(b));

    while (m != n) {
        if (m > n) {
            m -= n;
            m >>= u32(countTrailingZeros(m));
        } else {
            n -= m;
            n >>= u32(countTrailingZeros(n));
        }
    }

    return m << shift;
}

fn add_with_overflow(a: u32, b: u32) -> U32Overflow {
    let r = a + b;
    return U32Overflow(r, r < a);
}

fn mul_with_overflow(a: u32, b: u32) -> U32Overflow {
    let r = a * b;
    return U32Overflow(r, (b != 0u) && (r / b != a));
}

fn mul_fraction(a: Rational, b: Rational) -> Rational {
    if (a.den == 0u || b.den == 0u) {
        return OVERFLOW;
    }

    let gdc_ab: u32 = gcd(a.num, b.den);
    let gdc_ba: u32 = gcd(a.den, b.num);

    let an: u32 = a.num / gdc_ab;
    let bn: u32 = b.num / gdc_ba;
    let ad: u32 = a.den / gdc_ba;
    let bd: u32 = b.den / gdc_ab;

    let num = mul_with_overflow(an, bn);
    let den = mul_with_overflow(ad, bd);
    if (num.overflow || den.overflow) {
        return OVERFLOW;
    }

    return Rational(num.value, den.value, select(1u, 0u, a.sign != b.sign));
}

fn add_fraction(a: Rational, b: Rational) -> Rational {
    // Handle zeros
    if (a.num == 0u) {
       return b;
    }
    if (b.num == 0u) {
       return a;
    }

    let g = gcd(a.den, b.den);

    let an = mul_with_overflow(a.num, (b.den / g));
    let bn = mul_with_overflow(b.num, (a.den / g));

    if (an.overflow || bn.overflow) {
        return OVERFLOW;
    }

    var num: u32;
    var sign: u32;
    if (a.sign == b.sign) {
       let sum = add_with_overflow(an.value, bn.value);
       if (sum.overflow) {
           return OVERFLOW;
       }
       // Same sign, add the values
       num = sum.value;
       sign = a.sign;
    } else {
       // Different signs, subtract the smaller from the larger
       if (an.value > bn.value) {
           num = an.value - bn.value;
           sign = a.sign;
       } else if (bn.value > an.value) {
           num = bn.value - an.value;
           sign = b.sign;
       } else {
           // They are equal, result is zero
           return Rational(0u, 1u, 1u);
       }
    }

    // Final reduction
    // Common denominator without full lcm
    let den = mul_with_overflow(a.den / g, b.den);
    if (den.overflow) {
       return OVERFLOW;
    }

    let g2 = gcd(num, den.value);
    return Rational(num / g2, den.value / g2, sign);
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
        let product = mul_fraction(A[row * m + k], B[k * p + col]);

        if (product.den == 0u) {
            // Overflow detected in multiplication, set result to overflow value and break
            rational = OVERFLOW;
            break;
        }

        if (k == 0u) {
            rational = product;
        } else {
            rational = add_fraction(rational, product);

            if (rational.den == 0u) {
                // Overflow detected in addition, set result to overflow value and break
                rational = OVERFLOW;
                break;
            }
        }
    }

    // Store the result in c
    C[row * p + col] = rational;
}