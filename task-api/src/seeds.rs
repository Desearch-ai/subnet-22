//! Block numbers and the seeds their hashes give: a round's serve order and an upload's check draw come from a block nobody knew in advance.

use std::sync::Arc;

use anyhow::Result;
use desearch::time::now;

use crate::chain::Chain;
use crate::manifest::{local_block_hash, seed_from_hash, REVEAL_AFTER_BLOCKS};

pub const BLOCK_SECONDS: f64 = 12.0;

pub enum Seeds {
    /// Blocks counted on the clock from `genesis`, for tests and the devnet.
    Local {
        block_seconds: f64,
        genesis: f64,
    },
    Chain(Arc<Chain>),
}

impl Seeds {
    pub async fn current_block(&self) -> Result<i64> {
        match self {
            Seeds::Local { block_seconds, genesis } => Ok(((now() - genesis) / block_seconds).floor() as i64),
            Seeds::Chain(chain) => chain.current_block().await,
        }
    }

    /// The height every deadline is measured on: the chain's newest finalized block as last read, never counted on the clock.
    pub async fn height(&self) -> Result<i64> {
        match self {
            Seeds::Local { .. } => self.current_block().await,
            Seeds::Chain(chain) => chain.finalized_block().await,
        }
    }

    /// Seconds a block stands for when a duration is turned into blocks.
    pub fn block_seconds(&self) -> f64 {
        match self {
            Seeds::Local { block_seconds, .. } => *block_seconds,
            Seeds::Chain(_) => BLOCK_SECONDS,
        }
    }

    pub async fn target_block(&self) -> Result<i64> {
        Ok(self.current_block().await? + REVEAL_AFTER_BLOCKS)
    }

    pub fn wait_s(&self) -> f64 {
        match self {
            Seeds::Local { block_seconds, .. } => REVEAL_AFTER_BLOCKS as f64 * block_seconds,
            Seeds::Chain(_) => REVEAL_AFTER_BLOCKS as f64 * BLOCK_SECONDS,
        }
    }

    /// The hash of a block that exists, None before it does; the log keeps it beside every decision it seeds, so anyone can check it
    /// against a node of their own.
    pub async fn block_hash(&self, block: i64) -> Result<Option<String>> {
        if self.current_block().await? < block {
            return Ok(None);
        }
        Ok(match self {
            Seeds::Local { .. } => Some(local_block_hash(block)),
            Seeds::Chain(chain) => chain.block_hash(block).await?,
        })
    }

    /// The seed of a block that exists, None before it does.
    pub async fn seed_for(&self, block: i64) -> Result<Option<String>> {
        Ok(self.block_hash(block).await?.map(|hash| seed_from_hash(&hash)))
    }
}
