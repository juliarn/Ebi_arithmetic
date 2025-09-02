use crate::exact::MaybeExact;
use anyhow::Result;

pub trait EbiMatrix<T>:
    Clone + MaybeExact + IdentityMinus + GaussJordan + TryFrom<Vec<Vec<T>>> + Eq
where
    T: Clone,
{
    /// Creates a new matrix with each value initialised to zeroes.
    fn new(number_of_rows: usize, number_of_columns: usize) -> Self;

    /// Add a number of rows and columns to the matrix, initialised to zeroes.
    /// Does not decrease the size.
    fn increase_size_to(&mut self, number_of_rows: usize, number_of_columns: usize) {
        if self.number_of_columns() < number_of_columns {
            self.push_columns(number_of_columns - self.number_of_columns());
        }
        if self.number_of_rows() < number_of_rows {
            self.push_rows(number_of_rows - self.number_of_rows());
        }
    }

    /// Returns the number of rows
    fn number_of_rows(&self) -> usize;

    /// Returns the number of columns
    fn number_of_columns(&self) -> usize;

    /// Gets a particular value of the matrix, if it exists.
    fn get(&self, row: usize, column: usize) -> Option<T>;

    /// Returns whether a value of the matrix is one.
    /// If row and column do not exist, behaviour is undefined, and may panic.
    fn is_one(&self, row: usize, column: usize) -> bool;

    /// Returns whether a value of the matrix is larger than zero.
    /// If row and column do not exist, behaviour is undefined, and may panic.
    fn is_positive(&self, row: usize, column: usize) -> bool;

    /// Returns whether a value of the matrix is larger than zero.
    /// If row and column do not exist, behaviour is undefined, and may panic.
    fn is_negative(&self, row: usize, column: usize) -> bool;

    /// Sets a particular value of the matrix, if the row and column exist.
    /// If row and column do not exist, behaviour is undefined, and may panic.
    /// Prefer set_one and set_zero if possible.
    fn set(&mut self, row: usize, column: usize, value: T);

    /// Increases a particular value of the matrix, if the row and column exist.
    /// If row and column do not exist, behaviour is undefined, and may panic.
    fn increase(&mut self, row: usize, column: usize, value: &T);

    /// Decreases a particular value of the matrix, if the row and column exist.
    /// If row and column do not exist, behaviour is undefined, and may panic.
    fn decrease(&mut self, row: usize, column: usize, value: &T);

    /// Sets an entire row to zeroes.
    /// If row does not exist, behaviour is undefined, and may panic.
    fn set_row_zero(&mut self, row: usize);

    /// Sets a particular value of the matrix to zero, if the row and column exist.
    /// If row and column do not exist, behaviour is undefined, and may panic.
    fn set_zero(&mut self, row: usize, column: usize);

    /// Sets a particular value of the matrix to one, if the row and column exist.
    /// If row and column do not exist, behaviour is undefined, and may panic.
    fn set_one(&mut self, row: usize, column: usize);

    /// Adds a number of columns to the right side of the matrix.
    /// The added columns will be filled with zeroes.
    fn push_columns(&mut self, number_of_columns_to_add: usize);

    /// Adds a number of rows to the bottom of the matrix.
    /// The added rows will be filled with zeroes.
    fn push_rows(&mut self, number_of_rows_to_add: usize);

    /// Removes columsn from the left of the matrix.
    fn pop_front_columns(&mut self, number_of_columns_to_remove: usize);

    /// Returns a vector of the matrix
    fn to_vec(self) -> Vec<Vec<T>>;
}

pub trait IdentityMinus {
    /// For a given matrix M, computes I-M.
    /// The matrix does not need to be squared.
    fn identity_minus(&mut self);
}

pub trait Inversion {
    fn invert(self) -> Result<Self>
    where
        Self: Sized;
}

pub trait GaussJordan {
    /// Applies Gaussian elimination to obtain a matrix in row echelon form.
    fn gauss_jordan(&mut self);

    /// Applies Gaussian elimination to obtain a matrix in reduced row echelon form.
    fn gauss_jordan_reduced(self) -> Result<Self>
    where
        Self: Sized;
}