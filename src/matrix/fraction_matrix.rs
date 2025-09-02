//======================== set type alias based on compile flags ========================//
/// The fraction matrix is a matrix of fractions.
/// It applies two strategies to potentially save time for exact arithmetic matrices:
/// - it postpones reduction of fractions to the moment of export, or a user-chosen moment.
/// - it attempts to store values in primitives rather than BigUints at each reduction.
#[cfg(any(
    all(
        not(feature = "exactarithmetic"),
        not(feature = "approximatearithmetic")
    ),
    all(feature = "exactarithmetic", feature = "approximatearithmetic")
))]
pub type FractionMatrix = super::fraction_matrix_enum::FractionMatrixEnum;

#[cfg(all(not(feature = "exactarithmetic"), feature = "approximatearithmetic"))]
pub type FractionMatrix = super::fraction_matrix_f64::FractionMatrixF64;

#[cfg(all(feature = "exactarithmetic", not(feature = "approximatearithmetic")))]
pub type FractionMatrix = super::fraction_matrix_exact::FractionMatrixExact;

//======================== common code ========================//

#[macro_export]
macro_rules! push_columns {
    ($zero:expr, $number_of_columns_to_add:expr, $values:expr, $number_of_rows:expr, $number_of_columns:expr) => {
        for row in (0..$number_of_rows).rev() {
            $values.splice(
                row * $number_of_columns + $number_of_columns
                    ..row * $number_of_columns + $number_of_columns,
                vec![$zero; $number_of_columns_to_add],
            );
        }
    };
}

#[macro_export]
macro_rules! pop_front_columns {
    ($number_of_columns_to_remove:expr, $values:expr, $number_of_rows:expr, $number_of_columns:expr) => {
        for row in (0..$number_of_rows).rev() {
            $values.drain(
                row * $number_of_columns..row * $number_of_columns + $number_of_columns_to_remove,
            );
        }
    };
}

//======================== tests ========================//
#[cfg(test)]
mod tests {

    use crate::{
        Inversion,
        Zero,
        ebi_matrix::EbiMatrix,
        f, f0,
        fraction::fraction::Fraction,
        matrix::fraction_matrix::FractionMatrix,
    };

    #[test]
    fn fraction_matrix() {
        let m: FractionMatrix = vec![vec![f!(1, 4), f!(2, 5), f!(8, 3)]].try_into().unwrap();
        assert_eq!(m, m);
    }

    #[test]
    fn fraction_matrix_empty() {
        let m = vec![vec![]];
        let m: FractionMatrix = m.try_into().unwrap();
        assert_eq!(m.number_of_rows(), 1);
        assert_eq!(m.number_of_columns(), 0);
    }

    #[test]
    fn fraction_matrix_get() {
        let m: FractionMatrix = vec![vec![f!(1, 4), f!(2, 5), f!(8, 3)]].try_into().unwrap();

        assert!(m.get(10, 10).is_none());

        assert_eq!(m.get(0, 0).unwrap(), f!(1, 4));
    }

    #[test]
    #[should_panic]
    fn fraction_matrix_incomplete() {
        let _: FractionMatrix = vec![vec![f!(1, 4), f!(2, 5)], vec![f!(8, 3)]]
            .try_into()
            .unwrap();
    }

    #[test]
    fn fraction_matrix_pop_front() {
        let mut m1: FractionMatrix = vec![vec![f!(1, 4), f!(2, 5), f!(8, 3)]].try_into().unwrap();

        m1.pop_front_columns(1);

        let m3: FractionMatrix = vec![vec![f!(2, 5), f!(8, 3)]].try_into().unwrap();

        // println!("{:?}", m1);
        // println!("{:?}", m3);

        assert_eq!(m1, m3);
    }

    #[test]
    fn fraction_matrix_push_columns() {
        let mut m1: FractionMatrix = vec![vec![f!(1, 4), f!(2, 5), f!(8, 3)]].try_into().unwrap();

        m1.push_columns(1);

        let m3: FractionMatrix = vec![vec![f!(1, 4), f!(2, 5), f!(8, 3), f0!()]]
            .try_into()
            .unwrap();

        // println!("{:?}", m1);
        // println!("{:?}", m3);

        assert_eq!(m1, m3);
    }

    #[test]
    fn fraction_matrix_inverse() {
        let mut m1: FractionMatrix = vec![
            vec![1.into(), 0.into(), 0.into(), 0.into()],
            vec![0.into(), 1.into(), 0.into(), Fraction::from((-3, 5))],
            vec![0.into(), Fraction::from((-3, 4)), 1.into(), 0.into()],
            vec![0.into(), 0.into(), 0.into(), 1.into()],
        ]
        .try_into()
        .unwrap();

        let mut m2: FractionMatrix = vec![
            vec![1.into(), 0.into(), 0.into(), 0.into()],
            vec![0.into(), 1.into(), 0.into(), Fraction::from((3, 5))],
            vec![
                0.into(),
                Fraction::from((3, 4)),
                1.into(),
                Fraction::from((9, 20)),
            ],
            vec![0.into(), 0.into(), 0.into(), 1.into()],
        ]
        .try_into()
        .unwrap();

        m1 = m1.invert().unwrap();

        println!("{}", m1);
        println!("{}", m2);

        assert!(m1.eq(&mut m2));
    }

    #[test]
    fn display_empty() {
        let m = FractionMatrix::new(0, 0);
        let _ = format!("{}", m);

        let m = FractionMatrix::new(1, 0);
        let _ = format!("{}", m);

        let m = FractionMatrix::new(0, 1);
        let _ = format!("{}", m);
    }

    #[test]
    fn to_vec_empty() {
        let m = FractionMatrix::new(0, 0);
        let u: Vec<Vec<Fraction>> = vec![];
        assert_eq!(m.to_vec(), u);

        let m = FractionMatrix::new(1, 0);
        let u: Vec<Vec<Fraction>> = vec![vec![]];
        assert_eq!(m.to_vec(), u);

        let m = FractionMatrix::new(0, 1);
        let u: Vec<Vec<Fraction>> = vec![];
        assert_eq!(m.to_vec(), u);
    }
}
