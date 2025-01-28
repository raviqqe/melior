impl RegionLike<'c, 'a> {
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
    pub fn insert_block_after(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockAfter(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Inserts a block before another block.
    pub fn insert_block_before(&self, one: BlockRef<'c, '_>, other: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockBefore(self.raw, one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Appends a block.
    pub fn append_block(&self, block: Block<'c>) -> BlockRef<'c, '_> {
        unsafe {
            let r#ref = BlockRef::from_raw(block.to_raw());

            mlirRegionAppendOwnedBlock(self.raw, block.into_raw());

            r#ref
        }
    }

    /// Converts a region into a raw object.
    pub const fn into_raw(self) -> mlir_sys::MlirRegion {
        let region = self.raw;

        forget(self);

        region
    }
}
