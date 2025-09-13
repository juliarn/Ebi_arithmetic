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
    let gdc_ab: u32 = gcd(a.num, b.den);
    let gdc_ba: u32 = gcd(a.den, b.num);
    let num: u32 = (a.num / gdc_ab) * (b.num / gdc_ba);
    let den: u32 = (a.den / gdc_ba) * (b.den / gdc_ab);

    return Rational(num, den, select(u32(1u), u32(0u), a.sign != b.sign));
}

fn add_fraction(a: Rational,b: Rational) -> Rational {
   // Handle zeros
   if (a.num == u32(0u)) {
       return b;
   }
   if (b.num == u32(0u)) {
       return a;
   }

   let g = gcd(a.den, b.den);

   let an: u32 = a.num * (b.den / g);
   let bn: u32 = b.num * (a.den / g);

   var num: u32;
   var sign: u32;
   if (a.sign == b.sign) {
       // Same sign, add the values
       num = an + bn;
       sign = a.sign;
   } else {
       // Different signs, subtract the smaller from the larger
       if (an > bn) {
           num = an - bn;
           sign = a.sign;
       } else if (bn > an) {
           num = bn - an;
           sign = b.sign;
       } else {
           // They are equal, result is zero
           return Rational(u32(0u), u32(1u), u32(1u));
       }
   }

   // Final reduction
   // Common denominator without full lcm
   let den = (a.den / g) * b.den;
   let g2 = gcd(num, den);
   return Rational(num / g2, den / g2, sign);
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

    var rational = Rational(u32(0u), u32(1u), u32(1u));
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