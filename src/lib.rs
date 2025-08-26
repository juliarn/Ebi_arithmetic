pub mod choose_randomly;
pub mod ebi_number;
pub mod ebi_number_bigfraction;
pub mod ebi_number_bigint;
pub mod ebi_number_biguint;
pub mod ebi_number_floats;
pub mod ebi_number_integers;
pub mod exact;
pub mod fraction;
pub mod fraction_enum;
pub mod fraction_exact;
pub mod fraction_f64;
pub mod num;
pub mod parsing;

pub mod matrix {
    pub mod ebi_matrix;
    pub mod fraction_matrix;
    pub mod fraction_matrix_enum;
    pub mod fraction_matrix_exact;
    pub mod fraction_matrix_f64;
    pub mod inversion;

    pub mod identity_minus;
    pub mod loose_fraction;
    pub mod mul;
    pub mod mul_gpu;
    pub mod gauss_jordan;
}

pub mod fraction_raw {
    pub mod fraction_raw;
    pub mod getters;
    pub mod mul_assign;
    pub mod div_assign;
    pub mod recip;
    pub mod one;
    pub mod zero;
    pub mod neg;
    pub mod sub_assign;
}

pub mod shader {
    pub mod matrix_mul;
    pub mod state;
}
