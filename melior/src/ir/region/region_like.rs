use crate::ir::{Block, BlockRef};
use mlir_sys::{
    mlirRegionAppendOwnedBlock, mlirRegionGetFirstBlock, mlirRegionInsertOwnedBlockAfter,
    mlirRegionInsertOwnedBlockBefore, MlirRegion,
};
use std::mem::forget;

/// A region-like trait.
trait RegionLike<'c, 'a> {
    /// Converts a region into a raw object.
    fn to_raw(self) -> MlirRegion;

    /// Returns the first block in a region.
    fn first_block(&self) -> Option<BlockRef<'c, '_>> {
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
    fn insert_block_after(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockAfter(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Inserts a block before another block.
    fn insert_block_before(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockBefore(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Appends a block.
    fn append_block(&self, block: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(block.to_raw());

            mlirRegionAppendOwnedBlock(self.raw, block.into_raw());

            r#ref
        }
    }

    /// Converts a region into a raw object.
    fn into_raw(self) -> mlir_sys::MlirRegion {
        let region = self.raw;

        forget(self);

        region
    }
}
