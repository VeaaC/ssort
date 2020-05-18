use rand::distributions::Uniform;
use rand::prelude::*;

use std::mem::MaybeUninit;

struct Hole<T> {
    elem: MaybeUninit<T>,
    hole: *mut T,
}

unsafe fn shallow_clone<T>(value: &T) -> MaybeUninit<T> {
    MaybeUninit::new(std::ptr::read(value))
}

impl<T> Drop for Hole<T> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::copy_nonoverlapping(&*self.elem.as_ptr(), self.hole, 1);
        }
    }
}

pub fn insertion_sort<T: Ord>(data: &mut [T]) {
    unsafe {
        for i in 1..data.len() {
            let mut hole = Hole::<T> {
                elem: shallow_clone(&data[i]),
                hole: &mut data[i],
            };
            for j in (0..i).rev() {
                if data[j] > *hole.elem.as_ptr() {
                    std::ptr::copy_nonoverlapping(&data[j], &mut data[j + 1], 1);
                    hole.hole = &mut data[j];
                } else {
                    break;
                }
            }
        }
    }
}

pub fn move_sample_to_front<T: Ord>(data: &mut [T], sample_size: usize, rng: &mut impl Rng) {
    for i in 0..sample_size.min(data.len()) {
        let range = Uniform::from(i..data.len());
        data.swap(i, range.sample(rng));
    }
}

unsafe fn create_classification_tree<T: Ord + Eq>(
    sample: &mut [T],
    depth: usize,
    tree: &mut [MaybeUninit<T>],
) {
    debug_assert!(sample.len().next_power_of_two() == sample.len() + 1);
    debug_assert!((1 << depth) <= sample.len() + 1);

    sample.sort(); // TODO, replace with recursive call to sort

    let mut start = sample.len() / 2;
    let mut step = sample.len() + 1;
    let mut out = 0;
    for _ in 0..depth {
        for x in (start..sample.len()).step_by(step) {
            tree[out] = shallow_clone(&sample[x]);
            out += 1;
        }
        start /= 2;
        step /= 2;
    }
}

use u8 as BucketIndex;

fn classify<T: Ord>(
    data: &[T],
    depth: usize,
    tree: &[T],
    mut output: impl FnMut(usize, BucketIndex),
) {
    const CHUNK_SIZE: usize = 160;
    let num_buckets = 1 << depth;

    // First process unrolled chunks (hopefully using super-scalar parallelism)
    for start in (0..).step_by(CHUNK_SIZE) {
        if start + CHUNK_SIZE <= data.len() {
            let mut result: [BucketIndex; CHUNK_SIZE] = [0; CHUNK_SIZE];
            for _ in 0..depth {
                for x in 0..CHUNK_SIZE {
                    result[x] = (result[x] << 1)
                        + (data[start + x] >= tree[result[x] as usize]) as BucketIndex
                        + 1;
                }
            }
            for x in 0..CHUNK_SIZE {
                output(x + start, result[x] + 1 - num_buckets);
            }
        } else {
            // Then manually iterate over the rest
            for x in data.len() / CHUNK_SIZE * CHUNK_SIZE..data.len() {
                let mut result: BucketIndex = 0;
                for _ in 0..depth {
                    result =
                        dbg!((result << 1) + (data[x] >= tree[result as usize]) as BucketIndex + 1);
                }
                output(x, result + 1 - num_buckets);
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insertion_sort() {
        let mut data: Vec<u64> = vec![4, 1, 6, 8, 3, 7, 4, 7];
        insertion_sort(&mut data);
        assert_eq!(data, vec![1, 3, 4, 4, 6, 7, 7, 8]);
    }

    #[test]
    fn test_create_classification_tree() {
        let mut data: Vec<u64> = vec![4, 1, 6, 8, 3, 7, 4];
        let mut result: Vec<_> = std::iter::repeat(MaybeUninit::new(0)).take(7).collect();
        unsafe { create_classification_tree(&mut data, 1, &mut result) };
        assert_eq!(data, vec![1, 3, 4, 4, 6, 7, 8]);
        assert_eq!(
            unsafe { std::mem::transmute::<_, &[u64]>(&result[..]) },
            [4, 0, 0, 0, 0, 0, 0]
        );
        unsafe { create_classification_tree(&mut data, 2, &mut result) };
        assert_eq!(
            unsafe { std::mem::transmute::<_, &[u64]>(&result[..]) },
            [4, 3, 7, 0, 0, 0, 0]
        );
        unsafe { create_classification_tree(&mut data, 3, &mut result) };
        assert_eq!(
            unsafe { std::mem::transmute::<_, &[u64]>(&result[..]) },
            [4, 3, 7, 1, 4, 6, 8]
        );
    }

    #[test]
    fn test_classify() {
        let data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let tree = [4, 3, 7, 1, 4, 6, 8];
        let mut result = Vec::new();
        classify(&data, 3, &tree, |pos, bucket| {
            result.push((data[pos], bucket))
        });
        assert_eq!(
            result,
            vec![
                (0, 0),
                (1, 1),
                (2, 1),
                (3, 2),
                (4, 4),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 7),
                (0, 0),
                (1, 1),
                (2, 1),
                (3, 2),
                (4, 4),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 7),
            ]
        );
    }
}
