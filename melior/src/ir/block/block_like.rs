use super::{BlockArgument, BlockRef};
use crate::{
    ir::{operation::OperationRefMut, Location, Operation, OperationRef, RegionRef, Type, Value},
    Error,
};

pub trait BlockLike<'c, 'a> {
    /// Returns an argument at a position.
    fn argument(&self, index: usize) -> Result<BlockArgument<'c, 'a>, Error>;

    /// Returns a number of arguments.
    fn argument_count(&self) -> usize;

    /// Returns a reference to the first operation.
    fn first_operation(&self) -> Option<OperationRef<'c, 'a>>;
    /// Returns a mutable reference to the first operation.
    fn first_operation_mut(&mut self) -> Option<OperationRefMut<'c, 'a>>;

    /// Returns a reference to a terminator operation.
    fn terminator(&self) -> Option<OperationRef<'c, 'a>>;

    /// Returns a mutable reference to a terminator operation.
    fn terminator_mut(&mut self) -> Option<OperationRefMut<'c, 'a>>;

    /// Returns a parent region.
    // TODO Store lifetime of regions in blocks, or create another type like
    // `InsertedBlockRef`?
    fn parent_region(&self) -> Option<RegionRef<'c, 'a>>;

    /// Returns a parent operation.
    fn parent_operation(&self) -> Option<OperationRef<'c, 'a>>;

    /// Adds an argument.
    fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value<'c, 'a>;

    /// Appends an operation.
    fn append_operation(&self, operation: Operation<'c>) -> OperationRef<'c, 'a>;

    /// Inserts an operation.
    // TODO How can we make those update functions take `&mut self`?
    // TODO Use cells?
    fn insert_operation(&self, position: usize, operation: Operation<'c>) -> OperationRef<'c, 'a>;

    /// Inserts an operation after another.
    fn insert_operation_after(
        &self,
        one: OperationRef<'c, 'a>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'a>;

    /// Inserts an operation before another.
    fn insert_operation_before(
        &self,
        one: OperationRef<'c, 'a>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'a>;

    /// Returns a next block in a region.
    fn next_in_region(&self) -> Option<BlockRef<'c, 'a>>;
}
