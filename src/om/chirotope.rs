//! The chirotope of a configuration, evaluated lazily and memoised.

use std::collections::HashMap;
use std::fmt;
use std::sync::{PoisonError, RwLock};

use super::{OrientedMatroid, check_basis};
use crate::arith::Sign;
use crate::config::{Config, Kind, VectorConfig};
use crate::error::Result;
use crate::simplex::Simplex;

/// The chirotope of a full-rank configuration: `χ(B) = sign det(columns of B)`
/// over the homogenised matrix, with the columns in increasing label order.
///
/// Nothing is computed up front. Each sign is computed on first request with
/// [`Config::sign_of_simplex`] and cached, as TOPCOM's lazy `VirtualChiro`
/// does (`TOPCOM.md` section 2.2) — but only the sign is kept, not the
/// determinant, and the cache belongs to this value rather than to the process.
///
/// The configuration is copied in, in its homogenised form, so a chirotope has
/// no lifetime and does not care whether it came from points or vectors. It is
/// `Send + Sync`: the cache sits behind a lock, so one chirotope can be shared
/// by threads, and two chirotopes never share anything (`PLAN.md` section 5.3).
pub struct Chirotope {
    /// The homogenised configuration: a point configuration's columns with
    /// their trailing `1`, as vectors.
    vectors: VectorConfig,
    cache: RwLock<HashMap<Simplex, Sign>>,
}

impl Chirotope {
    /// The chirotope of `config`.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyConfiguration`](crate::Error::EmptyConfiguration) or
    /// [`Error::NotFullRank`](crate::Error::NotFullRank), from
    /// [`Config::require_full_rank`]. TOPCOM, and so the 0.3 package, rejects
    /// rank-deficient input; reducing it to its affine hull first is the
    /// caller's choice.
    pub fn new<K: Kind>(config: &Config<K>) -> Result<Self> {
        config.require_full_rank()?;
        let vectors =
            VectorConfig::from_row_major(config.homogenised(), config.len(), config.rows())?;
        Ok(Self {
            vectors,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// The number of signs computed so far.
    pub fn cached(&self) -> usize {
        self.cache
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

impl OrientedMatroid for Chirotope {
    fn n(&self) -> usize {
        self.vectors.len()
    }

    fn rank(&self) -> usize {
        self.vectors.rows()
    }

    fn sign(&self, basis: &Simplex) -> Result<Sign> {
        // A poisoned lock can only come from a panic elsewhere while holding
        // it; every write is a single insert, so the map is still consistent.
        if let Some(&s) = self
            .cache
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(basis)
        {
            return Ok(s);
        }
        check_basis(basis, self.n(), self.rank())?;
        let s = self.vectors.sign_of_simplex(basis)?;
        self.cache
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(basis.clone(), s);
        Ok(s)
    }
}

impl fmt::Debug for Chirotope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chirotope")
            .field("n", &self.n())
            .field("rank", &self.rank())
            .field("cached", &self.cached())
            .finish()
    }
}

impl<K: Kind> Config<K> {
    /// The chirotope of this configuration; see [`Chirotope::new`].
    ///
    /// # Errors
    ///
    /// Unless the configuration is non-empty and full rank.
    pub fn chirotope(&self) -> Result<Chirotope> {
        Chirotope::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use crate::config::PointConfig;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn is_send_and_sync() {
        assert_send_sync::<Chirotope>();
    }

    #[test]
    fn signs_are_memoised() {
        let tri = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1], 3, 2).unwrap();
        let chi = tri.chirotope().unwrap();
        assert_eq!((chi.n(), chi.rank(), chi.cached()), (3, 3, 0));
        let b = Simplex::from([0, 1, 2]);
        assert_eq!(chi.sign(&b), Ok(Sign::Positive));
        assert_eq!(chi.sign(&b), Ok(Sign::Positive));
        assert_eq!(chi.cached(), 1);
        // Malformed queries are errors and are not cached.
        assert!(chi.sign(&Simplex::from([0, 1])).is_err());
        assert!(chi.sign(&Simplex::from([0, 1, 3])).is_err());
        assert_eq!(chi.cached(), 1);
    }

    #[test]
    fn rank_deficient_configurations_are_rejected() {
        let line = PointConfig::from_row_major(&[0, 0, 1, 1, 2, 2], 3, 2).unwrap();
        assert_eq!(
            line.chirotope().err(),
            Some(Error::NotFullRank { rank: 2, rows: 3 })
        );
        assert_eq!(
            PointConfig::new().chirotope().err(),
            Some(Error::EmptyConfiguration)
        );
    }
}
