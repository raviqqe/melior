pub use super::{printing_flags::OperationPrintingFlags, result::OperationResult};
use super::{
    Attribute, AttributeLike, BlockRef, Identifier, Location, OperationRef, RegionRef, Value,
};
use crate::{context::ContextRef, utility::print_string_callback, Error, StringRef};
use mlir_sys::{
    mlirOperationDump, mlirOperationGetAttribute, mlirOperationGetAttributeByName,
    mlirOperationGetBlock, mlirOperationGetContext, mlirOperationGetLocation, mlirOperationGetName,
    mlirOperationGetNextInBlock, mlirOperationGetNumAttributes, mlirOperationGetNumOperands,
    mlirOperationGetNumRegions, mlirOperationGetNumResults, mlirOperationGetNumSuccessors,
    mlirOperationGetOperand, mlirOperationGetParentOperation, mlirOperationGetRegion,
    mlirOperationGetResult, mlirOperationGetSuccessor, mlirOperationPrintWithFlags,
    mlirOperationRemoveAttributeByName, mlirOperationRemoveFromParent,
    mlirOperationSetAttributeByName, mlirOperationVerify, MlirOperation,
};
use std::fmt::Display;

/// An operation-like  trait.
trait OperationLike<'c, 'a>: Copy + Display {
    /// Converts an operation into a raw object.
    fn to_raw(self) -> MlirOperation;

    /// Returns a context.
    pub fn context(self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.to_raw())) }
    }

    /// Returns a name.
    pub fn name(self) -> Identifier<'c> {
        unsafe { Identifier::from_raw(mlirOperationGetName(self.to_raw())) }
    }

    /// Returns a block.
    // TODO Store lifetime of block in operations, or create another type like
    // `AppendedOperationRef`?
    pub fn block(self) -> Option<BlockRef<'c, '_>> {
        unsafe { BlockRef::from_option_raw(mlirOperationGetBlock(self.to_raw())) }
    }

    /// Returns the number of operands.
    pub fn operand_count(self) -> usize {
        unsafe { mlirOperationGetNumOperands(self.to_raw()) as usize }
    }

    /// Returns the operand at a position.
    pub fn operand(self, index: usize) -> Result<Value<'c, '_>, Error> {
        if index < self.operand_count() {
            Ok(unsafe { Value::from_raw(mlirOperationGetOperand(self.to_raw(), index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "operation operand",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all operands.
    pub fn operands(self) -> impl Iterator<Item = Value<'c, '_>> {
        (0..self.operand_count()).map(|index| self.operand(index).expect("valid operand index"))
    }

    /// Returns the number of results.
    pub fn result_count(self) -> usize {
        unsafe { mlirOperationGetNumResults(self.to_raw()) as usize }
    }

    /// Returns a result at a position.
    pub fn result(self, index: usize) -> Result<OperationResult<'c, '_>, Error> {
        if index < self.result_count() {
            Ok(unsafe {
                OperationResult::from_raw(mlirOperationGetResult(self.to_raw(), index as isize))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "operation result",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all results.
    pub fn results(self) -> impl Iterator<Item = OperationResult<'c, '_>> {
        (0..self.result_count()).map(|index| self.result(index).expect("valid result index"))
    }

    /// Returns the number of regions.
    pub fn region_count(self) -> usize {
        unsafe { mlirOperationGetNumRegions(self.to_raw()) as usize }
    }

    /// Returns a region at a position.
    pub fn region(self, index: usize) -> Result<RegionRef<'c, '_>, Error> {
        if index < self.region_count() {
            Ok(unsafe {
                RegionRef::from_raw(mlirOperationGetRegion(self.to_raw(), index as isize))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "region",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all regions.
    pub fn regions(self) -> impl Iterator<Item = RegionRef<'c, 'a>> {
        (0..self.region_count()).map(|index| self.region(index).expect("valid result index"))
    }

    /// Gets the location of the operation.
    pub fn location(self) -> Location<'c> {
        unsafe { Location::from_raw(mlirOperationGetLocation(self.to_raw())) }
    }

    /// Returns the number of successors.
    pub fn successor_count(self) -> usize {
        unsafe { mlirOperationGetNumSuccessors(self.to_raw()) as usize }
    }

    /// Returns a successor at a position.
    pub fn successor(self, index: usize) -> Result<BlockRef<'c, 'a>, Error> {
        if index < self.successor_count() {
            Ok(unsafe {
                BlockRef::from_raw(mlirOperationGetSuccessor(self.to_raw(), index as isize))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "successor",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all successors.
    pub fn successors(self) -> impl Iterator<Item = BlockRef<'c, 'a>> {
        (0..self.successor_count())
            .map(|index| self.successor(index).expect("valid successor index"))
    }

    /// Returns the number of attributes.
    pub fn attribute_count(self) -> usize {
        unsafe { mlirOperationGetNumAttributes(self.to_raw()) as usize }
    }

    /// Returns a attribute at a position.
    pub fn attribute_at(self, index: usize) -> Result<(Identifier<'c>, Attribute<'c>), Error> {
        if index < self.attribute_count() {
            unsafe {
                let named_attribute = mlirOperationGetAttribute(self.to_raw(), index as isize);
                Ok((
                    Identifier::from_raw(named_attribute.name),
                    Attribute::from_raw(named_attribute.attribute),
                ))
            }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "attribute",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all attributes.
    pub fn attributes(self) -> impl Iterator<Item = (Identifier<'c>, Attribute<'c>)> + 'a {
        (0..self.attribute_count())
            .map(|index| self.attribute_at(index).expect("valid attribute index"))
    }

    /// Returns a attribute with the given name.
    pub fn attribute(self, name: &str) -> Result<Attribute<'c>, Error> {
        unsafe {
            Attribute::from_option_raw(mlirOperationGetAttributeByName(
                self.to_raw(),
                StringRef::new(name).to_raw(),
            ))
        }
        .ok_or_else(|| Error::AttributeNotFound(name.into()))
    }

    /// Checks if the operation has a attribute with the given name.
    pub fn has_attribute(self, name: &str) -> bool {
        self.attribute(name).is_ok()
    }

    /// Sets the attribute with the given name to the given attribute.
    pub fn set_attribute(&mut self, name: &str, attribute: Attribute<'c>) {
        unsafe {
            mlirOperationSetAttributeByName(
                self.to_raw(),
                StringRef::new(name).to_raw(),
                attribute.to_raw(),
            )
        }
    }

    /// Removes the attribute with the given name.
    pub fn remove_attribute(&mut self, name: &str) -> Result<(), Error> {
        unsafe { mlirOperationRemoveAttributeByName(self.to_raw(), StringRef::new(name).to_raw()) }
            .then_some(())
            .ok_or_else(|| Error::AttributeNotFound(name.into()))
    }

    /// Returns a reference to the next operation in the same block.
    pub fn next_in_block(self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(mlirOperationGetNextInBlock(self.to_raw())) }
    }

    /// Returns a mutable reference to the next operation in the same block.
    pub fn next_in_block_mut(self) -> Option<OperationRefMut<'c, 'a>> {
        unsafe { OperationRefMut::from_option_raw(mlirOperationGetNextInBlock(self.to_raw())) }
    }

    /// Returns a reference to the previous operation in the same block.
    pub fn previous_in_block(self) -> Option<OperationRef<'c, 'a>> {
        todo!("mlirOperationGetPrevInBlock is not exposed in the C API")
    }

    /// Returns a reference to a parent operation.
    pub fn parent_operation(self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(mlirOperationGetParentOperation(self.to_raw())) }
    }

    /// Removes itself from a parent block.
    pub fn remove_from_parent(&mut self) {
        unsafe { mlirOperationRemoveFromParent(self.to_raw()) }
    }

    /// Verifies an operation.
    pub fn verify(self) -> bool {
        unsafe { mlirOperationVerify(self.to_raw()) }
    }

    /// Dumps an operation.
    pub fn dump(self) {
        unsafe { mlirOperationDump(self.to_raw()) }
    }

    /// Prints an operation with flags.
    pub fn to_string_with_flags(self, flags: OperationPrintingFlags) -> Result<String, Error> {
        let mut data = (String::new(), Ok::<_, Error>(()));

        unsafe {
            mlirOperationPrintWithFlags(
                self.to_raw(),
                flags.to_raw(),
                Some(print_string_callback),
                &mut data as *mut _ as *mut _,
            );
        }

        data.1?;

        Ok(data.0)
    }
}
