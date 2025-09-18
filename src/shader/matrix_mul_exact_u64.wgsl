struct Rational {
    num: u64,
    den: u64,
    sign: u64,
};

struct Dimensions {
    n: u32,
    m: u32,
    p: u32,
};

struct U64Overflow {
    value: u64,
    overflow: bool,
}

@group(0) @binding(0)
var<storage, read> A: array<Rational>;

@group(0) @binding(1)
var<storage, read> B: array<Rational>;

@group(0) @binding(2)
var<uniform> dims: Dimensions;

@group(0) @binding(3)
var<storage, read_write> C: array<Rational>;

const OVERFLOW = Rational(u64(0xffffffffu), u64(0u), u64(1u));

fn gcd(a: u64, b: u64) -> u64 {
    if (a == u64(0u) || b == u64(0u)) {
        return (a | b);
    }

    let shift = u32(countTrailingZeros(a | b));

    var m = u64(a) >> u32(countTrailingZeros(a));
    var n = u64(b) >> u32(countTrailingZeros(b));

    while (m != n) {
        if (m > n) {
            m -= n;
            m >>= u32(countTrailingZeros(m));
        } else {
            n -= m;
            n >>= u32(countTrailingZeros(n));
        }
    }

    return u64(m << shift);
}

fn add_with_overflow(a: u64, b: u64) -> U64Overflow {
    let r = a + b;
    return U64Overflow(r, r < a);
}

fn mul_with_overflow(a: u64, b: u64) -> U64Overflow {
    let r = a * b;
    return U64Overflow(r, (b != u64(0u)) && (r / b != a));
}

fn mul_fraction(a: Rational, b: Rational) -> Rational {
    if (a.den == u64(0u) || b.den == u64(0u)) {
        return OVERFLOW;
    }

    let gdc_ab: u64 = gcd(a.num, b.den);
    let gdc_ba: u64 = gcd(a.den, b.num);

    let an: u64 = a.num / gdc_ab;
    let bn: u64 = b.num / gdc_ba;
    let ad: u64 = a.den / gdc_ba;
    let bd: u64 = b.den / gdc_ab;

    let num = mul_with_overflow(an, bn);
    let den = mul_with_overflow(ad, bd);
    if (num.overflow || den.overflow) {
        return OVERFLOW;
    }

    return Rational(num.value, den.value, select(u64(1u), u64(0u), a.sign != b.sign));
}

fn add_fraction(a: Rational,b: Rational) -> Rational {
   // Handle zeros
   if (a.num == u64(0u)) {
       return b;
   }
   if (b.num == u64(0u)) {
       return a;
   }

   let g = gcd(a.den, b.den);

   let an = mul_with_overflow(a.num, (b.den / g));
   let bn = mul_with_overflow(b.num, (a.den / g));

   if (an.overflow || bn.overflow) {
       return OVERFLOW;
   }

   var num: u64;
   var sign: u64;
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
           return Rational(u64(0u), u64(1u), u64(1u));
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

    var rational = Rational(u64(0u), u64(1u), u64(1u));
    for (var k = 0u; k < m; k++) {
        let product = mul_fraction(A[row * m + k], B[k * p + col]);

        if (product.den == u64(0u)) {
            // Overflow detected in multiplication, set result to overflow value and break
            rational = OVERFLOW;
            break;
        }

        if (k == 0u) {
            rational = product;
        } else {
            rational = add_fraction(rational, product);

            if (rational.den == u64(0u)) {
                // Overflow detected in addition, set result to overflow value and break
                rational = OVERFLOW;
                break;
            }
        }
    }

    // Store the result in c
    C[row * p + col] = rational;
}