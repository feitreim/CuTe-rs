use std::{fmt, iter::zip, vec::Vec};

// ------------------------------------------------------------------
// HTuple definiton and functions

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HTuple<T> {
    Leaf(T),
    Tuple(Vec<HTuple<T>>),
}

pub enum Slice {
    Idx(u64),
    Tuple(Vec<Slice>),
    All(),
}

pub type Shape = HTuple<u64>;
pub type Coord = HTuple<u64>;
pub type Stride = HTuple<i64>;

/// declare shapes like shape!(4, (3, 5))
macro_rules! shape {
    (($($inner:tt),+)) => { HTuple::Tuple(vec![$(shape!($inner)),+]) };
    ($val:expr)         => { HTuple::Leaf($val as u64) };
}

macro_rules! stride {
    (($($inner:tt),+)) => { HTuple::Tuple(vec![$(stride!($inner)),+]) };
    ($val:expr)         => { HTuple::Leaf($val as i64) };
}

macro_rules! slice {
    (_)                 => { Slice::All() };
    (($($inner:tt),+)) => { Slice::Tuple(vec![$(slice!($inner)),+]) };
    ($val:expr)         => { Slice::Idx($val as u64) };
}

impl<T> HTuple<T> {
    fn rank(&self) -> usize {
        match self {
            HTuple::Leaf(_) => 1,
            HTuple::Tuple(v) => v.len(),
        }
    }

    fn depth(&self) -> usize {
        match self {
            HTuple::Leaf(_) => 0,
            HTuple::Tuple(v) => 1 + v.iter().map(|x| x.depth()).max().unwrap_or(0),
        }
    }

    /// Element access X_i, will panic if out of range or on a leaf.
    fn get(&self, i: usize) -> &HTuple<T> {
        match self {
            HTuple::Leaf(_) => panic!("Cannot index into a leaf."),
            HTuple::Tuple(v) => &v[i],
        }
    }
}

// basic helper funcs
pub fn size(shape: &Shape) -> u64 {
    match shape {
        HTuple::Leaf(n) => *n,
        HTuple::Tuple(v) => v.iter().map(size).product(),
    }
}

pub fn congruent<A, B>(p: &HTuple<A>, s: &HTuple<B>) -> bool {
    match (p, s) {
        (HTuple::Leaf(_), HTuple::Leaf(_)) => true,
        (HTuple::Tuple(vp), HTuple::Tuple(vs)) => {
            vp.len() == vs.len() && vp.iter().zip(vs).all(|(a, b)| congruent(a, b))
        }
        _ => false,
    }
}

pub fn weakly_congruent<A, B>(p: &HTuple<A>, s: &HTuple<B>) -> bool {
    match (p, s) {
        (HTuple::Leaf(_), _) => true,
        (HTuple::Tuple(vp), HTuple::Tuple(vs)) => {
            vp.len() == vs.len() && vp.iter().zip(vs).all(|(a, b)| weakly_congruent(a, b))
        }
        _ => false,
    }
}

pub fn compatible(p: &Shape, s: &Shape) -> bool {
    match (p, s) {
        (HTuple::Leaf(n), _) => *n == size(s),
        (HTuple::Tuple(vp), HTuple::Tuple(vs)) => {
            vp.len() == vs.len() && vp.iter().zip(vs).all(|(a, b)| compatible(a, b))
        }
        _ => false,
    }
}

// conversion functions

pub fn idx2crd(idx: u64, shape: &Shape) -> Coord {
    match shape {
        HTuple::Leaf(_) => HTuple::Leaf(idx),
        HTuple::Tuple(modes) => {
            let mut rem = idx;
            let coords: Vec<_> = modes
                .iter()
                .map(|s_k| {
                    let sz = size(s_k);
                    let c = rem % sz;
                    rem /= sz;
                    idx2crd(c, s_k)
                })
                .collect();
            HTuple::Tuple(coords)
        }
    }
}

pub fn crd2idx(crd: &Coord, shape: &Shape) -> u64 {
    match (crd, shape) {
        (HTuple::Leaf(c), HTuple::Leaf(_)) => *c,
        (HTuple::Tuple(vc), HTuple::Tuple(vs)) => {
            let mut result = 0u64;
            let mut prod = 1u64;
            for (c_k, s_k) in zip(vc, vs) {
                result += crd2idx(c_k, s_k) * prod;
                prod *= size(s_k);
            }
            result
        }
        // weakly congruent case
        (HTuple::Leaf(c), HTuple::Tuple(_)) => *c,
        _ => panic!("incompatible coordinates for shape. (not weak cong.)"),
    }
}

/// inner product between a coordinate (HTuple<u64>) and a stride (HTuple<i64>)
pub fn inner_product(coord: &Coord, stride: &Stride) -> i64 {
    match (coord, stride) {
        (HTuple::Leaf(c), HTuple::Leaf(d)) => (*c as i64) * d,
        (HTuple::Tuple(vc), HTuple::Tuple(vs)) => {
            zip(vc, vs).map(|(ci, si)| inner_product(ci, si)).sum()
        }
        _ => panic!("Coordinate and Stride not congruent."),
    }
}

pub fn htuple2slice(coord: Coord) -> Slice {
    match coord {
        HTuple::Leaf(c) => Slice::Idx(c),
        HTuple::Tuple(v) => Slice::Tuple(v.into_iter().map(htuple2slice).collect()),
    }
}

// ------------------------------------------------------------------
// Layout definition and functions

#[derive(Debug, Clone)]
pub struct Layout {
    shape: Shape,
    stride: Stride,
}

impl Layout {
    pub fn new(shape: Shape, stride: Stride) -> Self {
        assert!(
            congruent(&shape, &stride),
            "shape and stride not congruent."
        );
        Layout { shape, stride }
    }

    /// evaluate layout for an integral coordinate
    pub fn call(&self, idx: u64) -> i64 {
        let nat_coord = idx2crd(idx, &self.shape);
        inner_product(&nat_coord, &self.stride)
    }

    /// evaluate layout for an arbitrary coordinate
    pub fn call_coord(&self, idx: &HTuple<u64>) -> i64 {
        let flat = crd2idx(idx, &self.shape);
        self.call(flat)
    }

    pub fn rank(&self) -> usize { self.shape.rank() }

    pub fn depth(&self) -> usize { self.shape.depth() }

    pub fn size(&self) -> u64 { size(&self.shape) }

    pub fn sublayout(&self, i: usize) -> Layout {
        match (&self.shape, &self.stride) {
            (HTuple::Tuple(sv), HTuple::Tuple(dv)) => Layout {
                shape: sv[i].clone(),
                stride: dv[i].clone(),
            },
            _ => panic!("cannot take sublayout of rank 1 layout."),
        }
    }

    pub fn slice(&self, args: &Slice) -> (i64, Layout) {
        match (args, &self.shape, &self.stride) {
            (Slice::All(), _, _) => (0, self.clone()),
            (Slice::Idx(i), HTuple::Leaf(_), HTuple::Leaf(s)) => {
                (*i as i64 * s, Layout::new(shape!(0), stride!(1)))
            }
            (Slice::Tuple(slices), HTuple::Tuple(shapes), HTuple::Tuple(strides)) => {
                let mut total_offset = 0i64;
                let mut res_shapes = Vec::new();
                let mut res_strides = Vec::new();

                for (c_k, (s_k, d_k)) in zip(slices, zip(shapes, strides)) {
                    let sub = Layout::new(s_k.clone(), d_k.clone());
                    let (off, residual) = sub.slice(c_k);
                    total_offset += off;

                    if size(&residual.shape) > 1 {
                        res_shapes.push(residual.shape);
                        res_strides.push(residual.stride);
                    }
                }

                let res_layout = match res_shapes.len() {
                    0 => Layout::new(shape!(1), stride!(0)),
                    1 => Layout::new(res_shapes.pop().unwrap(), res_strides.pop().unwrap()),
                    _ => Layout::new(HTuple::Tuple(res_shapes), HTuple::Tuple(res_strides)),
                };
                (total_offset, res_layout)
            }
            (Slice::Idx(c), HTuple::Tuple(_), _) => {
                let sl = htuple2slice(idx2crd(*c, &self.shape));
                self.slice(&sl)
            }
            _ => panic!("not weakly congruent."),
        }
    }
}

/// Flatten a layout into S:D pairs
pub fn flatten_layout(shape: &Shape, stride: &Stride) -> Vec<(u64, i64)> {
    match (shape, stride) {
        (HTuple::Leaf(s), HTuple::Leaf(d)) => vec![(*s, *d)],
        (HTuple::Tuple(sv), HTuple::Tuple(dv)) => zip(sv, dv)
            .flat_map(|(s, d)| flatten_layout(s, d))
            .collect(),
        _ => panic!("not congruent! (how did you get here)"),
    }
}

/// Assemble a vec of Layouts into a single Layout.
fn build_layout_from_layouts(layouts: Vec<Layout>) -> Layout {
    let shapes: Vec<_> = layouts.iter().map(|l| l.shape.clone()).collect();
    let strides: Vec<_> = layouts.iter().map(|l| l.stride.clone()).collect();

    match layouts.len() {
        0 => Layout::new(shape!(1), stride!(0)),
        1 => Layout::new(shapes[0].clone(), strides[0].clone()),
        _ => Layout::new(HTuple::Tuple(shapes), HTuple::Tuple(strides)),
    }
}

pub fn build_layout_from_pairs(pairs: Vec<(u64, i64)>) -> Layout {
    match pairs.len() {
        0 => Layout::new(shape!(1), stride!(0)),
        1 => Layout::new(HTuple::Leaf(pairs[0].0), HTuple::Leaf(pairs[0].1)),
        _ => Layout::new(
            HTuple::Tuple(pairs.iter().map(|(s, _)| HTuple::Leaf(*s)).collect()),
            HTuple::Tuple(pairs.iter().map(|(_, d)| HTuple::Leaf(*d)).collect()),
        ),
    }
}

/// Coalesce a layout to its simplest layout w/ minimal rank.
pub fn coalesce(layout: &Layout) -> Layout {
    let mut pairs = flatten_layout(&layout.shape, &layout.stride);
    // squeeze (like pytorch remove 1 dims)
    pairs.retain(|&(s, _)| s != 1);

    if pairs.is_empty() {
        return Layout::new(shape!(1), stride!(0));
    }

    //greedy merge the pairs back together
    let mut merged = vec![pairs[0]];
    for &(s, d) in &pairs[1..] {
        let last = merged.last_mut().unwrap();
        if last.0 as i64 * last.1 == d {
            // contiguous case so we extend the previous mode
            last.0 *= s;
        } else {
            merged.push((s, d));
        }
    }

    build_layout_from_pairs(merged)
}

/// Compose A ∘ (s : d) where A is already coalesced and d >= 0
fn simple_compose(a: &Layout, s: u64, d: u64) -> Layout {
    let pairs = flatten_layout(&a.shape, &a.stride);

    let mut result = Vec::new();
    let mut remaining_d = d;

    // Step 1: "Divide out" d from left to right
    for &(s_r, d_r) in &pairs {
        if remaining_d == 0 {
            // Past the division point, mode passes through unchanged
            result.push((s_r, d_r));
        } else if remaining_d >= s_r {
            // This mode is fully consumed by d
            // Check: d must be divisible by s_r
            assert!(remaining_d % s_r == 0, "stride divisibility violated");
            remaining_d /= s_r;
            // Mode collapses to size 1, effectively gone
        } else {
            // Partial consumption: d < s_r, so s_r must be divisible by d
            assert!(s_r % remaining_d == 0, "stride divisibility violated");
            let new_size = s_r / remaining_d;
            let new_stride = d_r * remaining_d as i64;
            remaining_d = 0;
            result.push((new_size, new_stride));
        }
    }

    result.retain(|&(s, _)| s != 1);

    if result.is_empty() {
        return Layout::new(shape!(1), stride!(0));
    };

    let current_size: u64 = result.iter().map(|(sh, _)| sh).product();
    if current_size != s {
        let prefix: u64 = result[..result.len() - 1]
            .iter()
            .map(|(sh, _)| sh)
            .product();
        assert!(s % prefix == 0, "shape divisibility violated");
        result.last_mut().unwrap().0 = s / prefix;
    }
    build_layout_from_pairs(result)
}

pub fn compose(lhs: &Layout, rhs: &Layout) -> Layout {
    let lhs = coalesce(lhs);

    match (&rhs.shape, &rhs.stride) {
        // base case, rank(B) = 1, depth(B) = 0
        (HTuple::Leaf(s), HTuple::Leaf(d)) => simple_compose(&lhs, *s, *d as u64),
        // distributive case, A ∘ B = A ∘ (B0, B1, ...) = (A ∘ B0, A ∘ B1, ...)
        (HTuple::Tuple(sv), HTuple::Tuple(dv)) => {
            let subs: Vec<_> = zip(sv, dv)
                .map(|(sk, dk)| {
                    let rhs_k = Layout::new(sk.clone(), dk.clone());
                    compose(&lhs, &rhs_k)
                })
                .collect();
            build_layout_from_layouts(subs)
        }
        _ => panic!("rhs incongruent with itself"),
    }
}

// print formatting

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Slice::Idx(v) => write!(f, "{v}"),
            Slice::All() => write!(f, ":"),
            Slice::Tuple(v) => {
                write!(f, "(")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{x}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl<T: fmt::Display> fmt::Display for HTuple<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HTuple::Leaf(v) => write!(f, "{v}"),
            HTuple::Tuple(v) => {
                write!(f, "(")?;
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{x}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = &self.shape;
        let stride = &self.stride;
        write!(f, "{shape} : {stride}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(n: u64) -> Shape { HTuple::Leaf(n) }
    fn tup(v: Vec<Shape>) -> Shape { HTuple::Tuple(v) }

    // -- Display --

    #[test]
    fn display_leaf() {
        let s = leaf(42);
        println!("{s}");
        assert_eq!(format!("{s}"), "42");
    }

    #[test]
    fn display_flat_tuple() {
        let s = tup(vec![leaf(3), leaf(4)]);
        println!("{s}");
        assert_eq!(format!("{s}"), "(3, 4)");
    }

    #[test]
    fn display_nested() {
        let s = tup(vec![tup(vec![leaf(2), leaf(3)]), leaf(5)]);
        println!("{s}");
        assert_eq!(format!("{s}"), "((2, 3), 5)");
    }

    // -- rank --

    #[test]
    fn rank_leaf() {
        let s = leaf(7);
        println!("rank({s}) = {}", s.rank());
        assert_eq!(s.rank(), 1);
    }

    #[test]
    fn rank_tuple() {
        let s = tup(vec![leaf(2), leaf(3), leaf(4)]);
        println!("rank({s}) = {}", s.rank());
        assert_eq!(s.rank(), 3);
    }

    // -- depth --

    #[test]
    fn depth_leaf() {
        let s = leaf(5);
        println!("depth({s}) = {}", s.depth());
        assert_eq!(s.depth(), 0);
    }

    #[test]
    fn depth_flat() {
        let s = tup(vec![leaf(2), leaf(3)]);
        println!("depth({s}) = {}", s.depth());
        assert_eq!(s.depth(), 1);
    }

    #[test]
    fn depth_nested() {
        let s = tup(vec![tup(vec![leaf(2), leaf(3)]), leaf(5)]);
        println!("depth({s}) = {}", s.depth());
        assert_eq!(s.depth(), 2);
    }

    #[test]
    fn depth_empty_tuple() {
        let s: Shape = tup(vec![]);
        println!("depth({s}) = {}", s.depth());
        assert_eq!(s.depth(), 1);
    }

    // -- get --

    #[test]
    fn get_element() {
        let s = tup(vec![leaf(10), leaf(20), leaf(30)]);
        println!("{s}[1] = {}", s.get(1));
        assert_eq!(s.get(1), &leaf(20));
    }

    // #[test]
    // #[should_panic(expected = "Cannot index into a leaf")]
    // fn get_leaf_panics() {
    //     leaf(5).get(0);
    // }

    // -- size --

    #[test]
    fn size_leaf() {
        let s = leaf(8);
        println!("size({s}) = {}", size(&s));
        assert_eq!(size(&s), 8);
    }

    #[test]
    fn size_flat() {
        let s = tup(vec![leaf(3), leaf(4)]);
        println!("size({s}) = {}", size(&s));
        assert_eq!(size(&s), 12);
    }

    #[test]
    fn size_nested() {
        // ((2, 3), 5) → 2*3*5 = 30
        let s = tup(vec![tup(vec![leaf(2), leaf(3)]), leaf(5)]);
        println!("size({s}) = {}", size(&s));
        assert_eq!(size(&s), 30);
    }

    #[test]
    fn size_empty_tuple() {
        let s: Shape = tup(vec![]);
        println!("size({s}) = {}", size(&s));
        assert_eq!(size(&s), 1); // empty product = 1
    }

    // -- congruent --

    #[test]
    fn congruent_leaves() {
        let a = leaf(3);
        let b = leaf(7);
        println!("congruent({a}, {b}) = {}", congruent(&a, &b));
        assert!(congruent(&a, &b));
    }

    #[test]
    fn congruent_matching_structure() {
        let a = tup(vec![leaf(2), leaf(3)]);
        let b = tup(vec![leaf(5), leaf(6)]);
        println!("congruent({a}, {b}) = {}", congruent(&a, &b));
        assert!(congruent(&a, &b));
    }

    #[test]
    fn congruent_mismatched_lengths() {
        let a = tup(vec![leaf(2), leaf(3)]);
        let b = tup(vec![leaf(5)]);
        println!("congruent({a}, {b}) = {}", congruent(&a, &b));
        assert!(!congruent(&a, &b));
    }

    #[test]
    fn congruent_leaf_vs_tuple() {
        let a = leaf(6);
        let b = tup(vec![leaf(2), leaf(3)]);
        println!("congruent({a}, {b}) = {}", congruent(&a, &b));
        assert!(!congruent(&a, &b));
    }

    // -- weakly_congruent --

    #[test]
    fn weakly_congruent_leaf_matches_anything() {
        let a = leaf(99);
        let b = tup(vec![leaf(2), leaf(3)]);
        println!("weakly_congruent({a}, {b}) = {}", weakly_congruent(&a, &b));
        assert!(weakly_congruent(&a, &b));
    }

    #[test]
    fn weakly_congruent_tuple_vs_leaf_fails() {
        let a = tup(vec![leaf(2), leaf(3)]);
        let b = leaf(6);
        println!("weakly_congruent({a}, {b}) = {}", weakly_congruent(&a, &b));
        assert!(!weakly_congruent(&a, &b));
    }

    #[test]
    fn weakly_congruent_nested() {
        // (leaf, (leaf, leaf)) weakly matches (tuple, tuple) if leaves match anything
        let a = tup(vec![leaf(1), tup(vec![leaf(2), leaf(3)])]);
        let b = tup(vec![
            tup(vec![leaf(4), leaf(5)]),
            tup(vec![leaf(6), leaf(7)]),
        ]);
        println!("weakly_congruent({a}, {b}) = {}", weakly_congruent(&a, &b));
        assert!(weakly_congruent(&a, &b));
    }

    // -- compatible --

    #[test]
    fn compatible_leaf_matches_size() {
        let p = leaf(6);
        let s = tup(vec![leaf(2), leaf(3)]);
        println!("compatible({p}, {s}) = {}", compatible(&p, &s));
        assert!(compatible(&p, &s));
    }

    #[test]
    fn compatible_leaf_wrong_size() {
        let p = leaf(5);
        let s = tup(vec![leaf(2), leaf(3)]);
        println!("compatible({p}, {s}) = {}", compatible(&p, &s));
        assert!(!compatible(&p, &s));
    }

    #[test]
    fn compatible_matching_tuples() {
        let p = tup(vec![leaf(6), leaf(10)]);
        let s = tup(vec![
            tup(vec![leaf(2), leaf(3)]),
            tup(vec![leaf(2), leaf(5)]),
        ]);
        println!("compatible({p}, {s}) = {}", compatible(&p, &s));
        assert!(compatible(&p, &s));
    }

    #[test]
    fn compatible_tuple_vs_leaf_fails() {
        let p = tup(vec![leaf(2), leaf(3)]);
        let s = leaf(6);
        println!("compatible({p}, {s}) = {}", compatible(&p, &s));
        assert!(!compatible(&p, &s));
    }

    // -- Layout display --

    #[test]
    fn display_layout() {
        let layout = Layout {
            shape: tup(vec![leaf(2), leaf(3)]),
            stride: HTuple::Tuple(vec![HTuple::Leaf(1), HTuple::Leaf(2)]),
        };
        println!("{layout}");
        assert_eq!(format!("{layout}"), "(2, 3) : (1, 2)");
    }

    // -- Slice display --

    #[test]
    fn display_slice() {
        let s = slice!((_, 2));
        println!("{s}");
        assert_eq!(format!("{s}"), "(:, 2)");
    }

    #[test]
    fn display_slice_nested() {
        let s = slice!(((_, 3), _));
        println!("{s}");
        assert_eq!(format!("{s}"), "((:, 3), :)");
    }

    // -- slice --

    #[test]
    fn slice_row_from_rowmajor() {
        // (4,3):(3,1) — 4x3 row-major, fix row=2 → offset=6, column layout 3:1
        let layout = Layout::new(shape!((4, 3)), stride!((3, 1)));
        let (off, sub) = layout.slice(&slice!((2, _)));
        println!("slice({layout}, (2, :)) = off={off}, {sub}");
        assert_eq!(off, 6);
        assert_eq!(format!("{sub}"), "3 : 1");
    }

    #[test]
    fn slice_col_from_rowmajor() {
        // (4,3):(3,1) — fix col=1 → offset=1, row layout 4:3
        let layout = Layout::new(shape!((4, 3)), stride!((3, 1)));
        let (off, sub) = layout.slice(&slice!((_, 1)));
        println!("slice({layout}, (:, 1)) = off={off}, {sub}");
        assert_eq!(off, 1);
        assert_eq!(format!("{sub}"), "4 : 3");
    }

    #[test]
    fn slice_row_from_colmajor() {
        // (4,3):(1,4) — 4x3 col-major, fix row=2 → offset=2, col layout 3:4
        let layout = Layout::new(shape!((4, 3)), stride!((1, 4)));
        let (off, sub) = layout.slice(&slice!((2, _)));
        println!("slice({layout}, (2, :)) = off={off}, {sub}");
        assert_eq!(off, 2);
        assert_eq!(format!("{sub}"), "3 : 4");
    }

    #[test]
    fn slice_all_preserves_layout() {
        let layout = Layout::new(shape!((4, 3)), stride!((3, 1)));
        let (off, sub) = layout.slice(&slice!(_));
        println!("slice({layout}, :) = off={off}, {sub}");
        assert_eq!(off, 0);
        assert_eq!(format!("{sub}"), format!("{layout}"));
    }

    #[test]
    fn slice_flat_idx_into_tuple() {
        // (4,3):(3,1) — flat idx 5 → coord (1,1) → offset 1*3+1*1=4
        let layout = Layout::new(shape!((4, 3)), stride!((3, 1)));
        let (off, sub) = layout.slice(&slice!(5));
        println!("slice({layout}, 5) = off={off}, {sub}");
        assert_eq!(off, 4);
    }

    #[test]
    fn slice_nested_shape() {
        // ((2,4), 3):((1,2), 8) — fix mode-1 to 1, keep nested mode-0
        let layout = Layout::new(shape!(((2, 4), 3)), stride!(((1, 2), 8)));
        let (off, sub) = layout.slice(&slice!((_, 1)));
        println!("slice({layout}, (:, 1)) = off={off}, {sub}");
        assert_eq!(off, 8);
        assert_eq!(format!("{sub}"), "(2, 4) : (1, 2)");
    }

    // -- coalesce --

    #[test]
    fn coalesce_contiguous() {
        // (4, 8):(1, 4) → 32:1 — modes are contiguous (4*1 == 4)
        let layout = Layout::new(shape!((4, 8)), stride!((1, 4)));
        let c = coalesce(&layout);
        println!("coalesce({layout}) = {c}");
        assert_eq!(format!("{c}"), "32 : 1");
    }

    #[test]
    fn coalesce_non_contiguous() {
        // (4, 8):(1, 8) — gap between modes (4*1 != 8), no merge
        let layout = Layout::new(shape!((4, 8)), stride!((1, 8)));
        let c = coalesce(&layout);
        println!("coalesce({layout}) = {c}");
        assert_eq!(format!("{c}"), "(4, 8) : (1, 8)");
    }

    #[test]
    fn coalesce_removes_unit_dims() {
        // (1, 4, 8):(5, 1, 4) → squeeze unit dim, then 4 and 8 are contiguous
        let layout = Layout::new(shape!((1, 4, 8)), stride!((5, 1, 4)));
        let c = coalesce(&layout);
        println!("coalesce({layout}) = {c}");
        assert_eq!(format!("{c}"), "32 : 1");
    }

    #[test]
    fn coalesce_all_unit_dims() {
        // (1, 1, 1):(3, 5, 7) → everything squeezed
        let layout = Layout::new(shape!((1, 1, 1)), stride!((3, 5, 7)));
        let c = coalesce(&layout);
        println!("coalesce({layout}) = {c}");
        assert_eq!(format!("{c}"), "1 : 0");
    }

    #[test]
    fn coalesce_nested_contiguous() {
        // ((2, 4), 8):((1, 2), 8) — flattens to (2,4,8):(1,2,8), all contiguous → 64:1
        let layout = Layout::new(shape!(((2, 4), 8)), stride!(((1, 2), 8)));
        let c = coalesce(&layout);
        println!("coalesce({layout}) = {c}");
        assert_eq!(format!("{c}"), "64 : 1");
    }

    #[test]
    fn coalesce_partial_merge() {
        // (2, 4, 3):(1, 2, 16) — first two contiguous (2*1==2), third is not (8!=16)
        let layout = Layout::new(shape!((2, 4, 3)), stride!((1, 2, 16)));
        let c = coalesce(&layout);
        println!("coalesce({layout}) = {c}");
        assert_eq!(format!("{c}"), "(8, 3) : (1, 16)");
    }

    #[test]
    fn coalesce_already_minimal() {
        // single mode, nothing to merge
        let layout = Layout::new(shape!(16), stride!(2));
        let c = coalesce(&layout);
        println!("coalesce({layout}) = {c}");
        assert_eq!(format!("{c}"), "16 : 2");
    }

    // -- compose --

    #[test]
    fn compose_leaf_leaf_unit_stride() {
        // compose(8:2, 4:1) = 4:2 — take first 4 elements, preserve stride
        let a = Layout::new(shape!(8), stride!(2));
        let b = Layout::new(shape!(4), stride!(1));
        let c = compose(&a, &b);
        println!("compose({a}, {b}) = {c}");
        assert_eq!(format!("{c}"), "4 : 2");
    }

    #[test]
    fn compose_leaf_leaf_strided() {
        // compose(8:2, 2:4) = 2:8 — every 4th element of A, strides multiply
        let a = Layout::new(shape!(8), stride!(2));
        let b = Layout::new(shape!(2), stride!(4));
        let c = compose(&a, &b);
        println!("compose({a}, {b}) = {c}");
        assert_eq!(format!("{c}"), "2 : 8");
    }

    #[test]
    fn compose_multimodal_a_skip_mode() {
        // compose((4,3):(1,4), 3:4) — B steps by 4, skips mode-0 entirely → 3:4
        let a = Layout::new(shape!((4, 3)), stride!((1, 4)));
        let b = Layout::new(shape!(3), stride!(4));
        let c = compose(&a, &b);
        println!("compose({a}, {b}) = {c}");
        assert_eq!(format!("{c}"), "3 : 4");
    }

    #[test]
    fn compose_multimodal_a_within_mode() {
        // compose((4,3):(1,4), 2:1) — take first 2 from mode-0 → 2:1
        let a = Layout::new(shape!((4, 3)), stride!((1, 4)));
        let b = Layout::new(shape!(2), stride!(1));
        let c = compose(&a, &b);
        println!("compose({a}, {b}) = {c}");
        assert_eq!(format!("{c}"), "2 : 1");
    }

    #[test]
    fn compose_leaf_a_multimodal_b() {
        // compose(32:2, (4,8):(1,4)) = (4,8):(2,8) — distributes, strides scale by 2
        let a = Layout::new(shape!(32), stride!(2));
        let b = Layout::new(shape!((4, 8)), stride!((1, 4)));
        let c = compose(&a, &b);
        println!("compose({a}, {b}) = {c}");
        assert_eq!(format!("{c}"), "(4, 8) : (2, 8)");
    }

    #[test]
    fn compose_identity_tiling() {
        // compose(32:1, (4,8):(1,4)) = (4,8):(1,4) — contiguous A with tiled B
        let a = Layout::new(shape!(32), stride!(1));
        let b = Layout::new(shape!((4, 8)), stride!((1, 4)));
        let c = compose(&a, &b);
        println!("compose({a}, {b}) = {c}");
        assert_eq!(format!("{c}"), "(4, 8) : (1, 4)");
    }

    #[test]
    fn compose_nested_a_coalesced() {
        // ((2,4),8):((1,2),8) coalesces to 64:1, then compose with 4:1 → 4:1
        let a = Layout::new(shape!(((2, 4), 8)), stride!(((1, 2), 8)));
        let b = Layout::new(shape!(4), stride!(1));
        let c = compose(&a, &b);
        println!("compose({a}, {b}) = {c}");
        assert_eq!(format!("{c}"), "4 : 1");
    }

    #[test]
    fn compose_brute_force() {
        // verify compose pointwise: compose(A, B)(i) == A(B(i))
        let a = Layout::new(shape!((4, 3)), stride!((1, 4)));
        let b = Layout::new(shape!((2, 3)), stride!((1, 2)));
        let c = compose(&a, &b);
        println!("compose({a}, {b}) = {c}");
        for i in 0..c.size() {
            let expected = a.call(b.call(i) as u64);
            let actual = c.call(i);
            assert_eq!(
                actual, expected,
                "mismatch at i={i}: compose={actual}, A(B({i}))={expected}"
            );
        }
    }
}
