use super::{Block, BlockRef};
use mlir_sys::{
    mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegionDestroy, mlirRegionEqual,
    mlirRegionGetFirstBlock, mlirRegionInsertOwnedBlockAfter, mlirRegionInsertOwnedBlockBefore,
    MlirRegion,
};
use std::{
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};

/// A region.
#[derive(Debug)]
pub struct Region<'c> {
    raw: MlirRegion,
    _block: PhantomData<Block<'c>>,
}

impl<'c> Region<'c> {
    /// Creates a region.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirRegionCreate() },
            _block: Default::default(),
        }
    }

    /// Returns the first block in a region.
    pub fn first_block(&self) -> Option<BlockRef<'c, '_>> {
        unsafe {
            let block = mlirRegionGetFirstBlock(self.raw);

            if block.ptr.is_null() {
                None
            } else {
                Some(BlockRef::from_raw(block))
            }
        }
    }

    /// Inserts a block after another block.
<<<<<<< Updated upstream
    pub fn insert_block_after(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef<'c, '_> {
=======
    pub fn insert_block_after(&mut self, one: BlockRef<'c, '_>, other: Block<'c>) {
>>>>>>> Stashed changes
        unsafe {
            mlirRegionInsertOwnedBlockAfter(self.raw, one.to_raw(), other.into_raw());
        }
    }

    /// Inserts a block before another block.
<<<<<<< Updated upstream
    pub fn insert_block_before(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef<'c, '_> {
=======
    pub fn insert_block_before(&mut self, one: BlockRef<'c, '_>, other: Block<'c>) {
>>>>>>> Stashed changes
        unsafe {
            mlirRegionInsertOwnedBlockBefore(self.raw, one.to_raw(), other.into_raw());
        }
    }

    /// Appends a block.
<<<<<<< Updated upstream
    pub fn append_block(&self, block: Block<'c>) -> BlockRef<'c, '_> {
=======
    pub fn append_block(&mut self, block: Block<'c>) {
>>>>>>> Stashed changes
        unsafe {
            mlirRegionAppendOwnedBlock(self.raw, block.into_raw());
        }
    }

    /// Converts a region into a raw object.
    pub fn into_raw(self) -> mlir_sys::MlirRegion {
        let region = self.raw;

        forget(self);

        region
    }
}

impl<'c> Default for Region<'c> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'c> Drop for Region<'c> {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.raw) }
    }
}

impl<'c> PartialEq for Region<'c> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl<'c> Eq for Region<'c> {}

/// A reference to a region.
#[derive(Clone, Copy, Debug)]
pub struct RegionRef<'c, 'a> {
    raw: MlirRegion,
    _region: PhantomData<&'a Region<'c>>,
}

impl<'c, 'a> RegionRef<'c, 'a> {
    /// Creates a region from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirRegion) -> Self {
        Self {
            raw,
            _region: Default::default(),
        }
    }

    /// Creates an optional region from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirRegion) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'c, 'a> Deref for RegionRef<'c, 'a> {
    type Target = Region<'c>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl<'c, 'a> PartialEq for RegionRef<'c, 'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl<'c, 'a> Eq for RegionRef<'c, 'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::Location, ir::Type, test::create_test_context};

    #[test]
    fn new() {
        Region::new();
    }

    #[test]
    fn first_block() {
        assert!(Region::new().first_block().is_none());
    }

    #[test]
    fn append_block() {
        let mut region = Region::new();
        let block = Block::new(&[]);

        region.append_block(block);

        assert!(region.first_block().is_some());
    }

    #[test]
    fn insert_block_after() {
        let context = create_test_context();
        let mut region = Region::new();

        region.append_block(Block::new(&[]));
        region.insert_block_after(
            region.first_block().unwrap(),
            Block::new(&[(Type::index(&context), Location::unknown(&context))]),
        );

        assert_eq!(region.first_block().unwrap().argument_count(), 0);
    }

    #[test]
    fn insert_block_before() {
        let context = create_test_context();
        let mut region = Region::new();

        region.append_block(Block::new(&[]));
        region.insert_block_before(
            region.first_block().unwrap(),
            Block::new(&[(Type::index(&context), Location::unknown(&context))]),
        );

        assert_eq!(region.first_block().unwrap().argument_count(), 1);
    }

    #[test]
    fn equal() {
        let region = Region::new();

        assert_eq!(region, region);
    }

    #[test]
    fn not_equal() {
        assert_ne!(Region::new(), Region::new());
    }
}
