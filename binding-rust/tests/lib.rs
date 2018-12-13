extern crate gym;

use gym::*;

const NUM_SAMPLES: usize = 25;

#[test]
fn test_discrete_space_sample() {
	let discrete_space = Space::DISCRETE{n: 15};
	for _ in 0..NUM_SAMPLES {
		let sample = discrete_space.sample();
		assert!(sample.len() == 1 && 0. <= sample[0] && sample[0] < 15.);
	}
}

#[test]
fn test_box_space_sample() {
	let box_space = Space::BOX{
		shape: vec![5], 
		high: vec![1., 2., 3., 4., 5.], 
		low: vec![-1., -2., -3., -4., -5.]
	};
	for _ in 0..NUM_SAMPLES {
		let sample = box_space.sample();
		assert_eq!(sample.len(), 5);
		for i in 0..5 {
			let bound = (i+1) as f64;
			assert!(-bound <= sample[i] && sample[i] <= bound);
		}
	}
}

#[test]
fn test_big_box_space_sample() {
	let box_space = Space::BOX {
		shape: vec![3, 2, 2], 
		high: vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], 
		low: vec![-1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.]
	};
	for _ in 0..NUM_SAMPLES {
		let sample = box_space.sample();
		assert_eq!(sample.len(), 12);
		for i in 0..12 {
			let bound = (i+1) as f64;
			assert!(-bound <= sample[i] && sample[i] <= bound);
		}
	}
}

#[test]
fn test_tuple_space_sample() {
	let discrete_space = Space::DISCRETE{n: 15};
	let box_space = Space::BOX {
		shape: vec![5], 
		high: vec![1., 2., 3., 4., 5.], 
		low: vec![-1., -2., -3., -4., -5.]
	};

	let tuple_space = Space::TUPLE {
		spaces: vec![
			Box::new(discrete_space), 
			Box::new(box_space)
		]
	};
	for _ in 0..NUM_SAMPLES {
		let sample = tuple_space.sample();
		assert_eq!(sample.len(), 6);
		assert!(0. <= sample[0] && sample[0] <= 15.);
		for i in 1..6 {
			let bound = i as f64;
			assert!(-bound <= sample[i] && sample[i] <= bound);
		}
	}
}